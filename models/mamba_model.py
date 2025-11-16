import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict, Any, Optional, Tuple
from einops import rearrange


class SelectiveScan(nn.Module):
    """
    选择性扫描机制

    实现 Mamba 模型的核心选择性状态空间扫描算法，
    允许模型根据输入内容动态调整状态转移矩阵

    Args:
        d_model (int): 模型维度
        d_state (int): 状态空间维度，默认为16
        d_conv (int): 卷积核大小，默认为4
        expand (int): 扩展因子，默认为2

    Example:
        >>> scan = SelectiveScan(d_model=256, d_state=16)
        >>> output = scan(input_tensor)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # 输入投影层
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 卷积层用于局部依赖
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # 激活函数
        self.activation = "silu"
        self.act = nn.SiLU()

        # SSM 参数投影
        self.x_proj = nn.Linear(self.d_inner,
                                self.d_inner + self.d_state * 2,
                                bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # 状态空间参数
        A_log = torch.log(torch.arange(1, self.d_state + 1,
                                       dtype=torch.float32).repeat(
                                           self.d_inner, 1))
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 输入投影
        xz = self.in_proj(x)  # (batch_size, seq_len, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # 分割为 x 和 z

        # 卷积处理
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d l -> b l d')

        # 激活
        x = self.act(x)

        # 计算 SSM 参数
        x_dbl = self.x_proj(x)  # (batch_size, seq_len, d_inner + d_state * 2)
        delta, B, C = torch.split(
            x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))

        # 状态空间计算
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        y = self._selective_scan(x, delta, A, B, C, self.D.float())

        # 门控机制
        y = y * self.act(z)

        # 输出投影
        output = self.out_proj(y)

        return output

    def _selective_scan(self, u: torch.Tensor, delta: torch.Tensor,
                        A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                        D: torch.Tensor) -> torch.Tensor:
        """
        选择性扫描核心算法

        实现状态空间模型的递归计算，支持选择性状态更新

        Args:
            u: 输入序列
            delta: 时间步长参数
            A: 状态转移矩阵
            B: 输入矩阵
            C: 输出矩阵
            D: 直连权重

        Returns:
            torch.Tensor: 扫描输出结果
        """
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[-1]

        # 离散化参数
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (batch_size, seq_len,
        #                                              d_inner, d_state)
        deltaB_u = (delta.unsqueeze(-1) * B.unsqueeze(2) *
                    u.unsqueeze(-1))  # (batch_size, seq_len, d_inner, d_state)

        # 初始化状态
        x = torch.zeros(batch_size, d_inner, d_state, device=u.device,
                        dtype=u.dtype)
        ys = []

        # 递归计算
        for i in range(seq_len):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.einsum('bnd,bd->bn', x, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (batch_size, seq_len, d_inner)
        y = y + u * D

        return y


class MambaBlock(nn.Module):
    """
    Mamba 基础块

    包含选择性扫描机制和残差连接的完整 Mamba 块

    Args:
        d_model (int): 模型维度
        d_state (int): 状态空间维度
        d_conv (int): 卷积核大小
        expand (int): 扩展因子

    Example:
        >>> block = MambaBlock(d_model=256)
        >>> output = block(input_tensor)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.d_model = d_model

        # 层归一化
        self.norm = nn.LayerNorm(d_model)

        # 选择性扫描层
        self.mixer = SelectiveScan(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 输出张量
        """
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        return residual + x


class CipherMamba(nn.Module):
    """
    用于密文预测/明文恢复的 Mamba 模型

    基于 Mamba 架构的序列到序列模型，专门用于密码学分析任务

    Args:
        input_dim (int): 输入特征维度，通常为64（16位明文/密文）
        hidden_dim (int): 隐藏层维度，默认为256
        num_layers (int): Mamba 层数，默认为4
        output_dim (int): 输出维度，通常为8（8位目标位）
        d_state (int): 状态空间维度，默认为16
        d_conv (int): 卷积核大小，默认为4
        expand (int): 扩展因子，默认为2
        dropout (float): Dropout 概率，默认为0.1

    Example:
        >>> model = CipherMamba(input_dim=16, hidden_dim=256,
        ...                     num_layers=4, output_dim=16)
        >>> model.saveModel('checkpoint.pth', optimizer, epoch=10, loss=0.1)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 4, output_dim: int = 16,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout_rate = dropout

        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Mamba 层堆叠
        self.layers = nn.ModuleList([
            MambaBlock(hidden_dim, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])

        # 输出层
        self.norm_f = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # 输出0/1概率

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, seq_len, output_dim)
        """
        # 输入嵌入
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)

        # 通过 Mamba 层
        for layer in self.layers:
            x = layer(x)

        # 最终处理 - 保持序列维度
        x = self.norm_f(x)  # (batch_size, seq_len, hidden_dim)
        x = self.dropout(x)
        x = self.head(x)  # (batch_size, seq_len, output_dim)

        return self.sigmoid(x)

    def saveModel(self, filepath: str,
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  epoch: int = 0, loss: float = 0.0,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        保存模型检查点

        保存模型状态字典、优化器状态、训练元数据等信息到指定文件

        Args:
            filepath (str): 保存文件路径
            optimizer (torch.optim.Optimizer, optional): 优化器实例
            epoch (int): 当前训练轮数
            loss (float): 当前损失值
            metadata (Dict[str, Any], optional): 额外的元数据信息

        Raises:
            OSError: 文件保存失败时抛出异常

        Example:
            >>> model = CipherMamba(16, 256, 4, 16)
            >>> model.saveModel('checkpoint.pth', optimizer,
            ...                 epoch=10, loss=0.1)
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'output_dim': self.output_dim,
                'd_state': self.d_state,
                'd_conv': self.d_conv,
                'expand': self.expand,
                'dropout': self.dropout_rate
            },
            'epoch': epoch,
            'loss': loss
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if metadata is not None:
            checkpoint['metadata'] = metadata

        torch.save(checkpoint, filepath)
        print(f"模型已保存至: {filepath}")

    @classmethod
    def loadModel(cls, filepath: str,
                  device: str = 'cpu') -> Tuple['CipherMamba',
                                                Dict[str, Any]]:
        """
        从检查点文件加载模型

        从保存的检查点文件中恢复模型实例和相关信息

        Args:
            filepath (str): 检查点文件路径
            device (str): 设备类型 ('cpu' 或 'cuda')

        Returns:
            Tuple[CipherMamba, Dict[str, Any]]: (模型实例, 检查点信息字典)

        Raises:
            FileNotFoundError: 检查点文件不存在时抛出异常
            KeyError: 检查点文件格式错误时抛出异常

        Example:
            >>> model, checkpoint = CipherMamba.loadModel('checkpoint.pth')
            >>> print(f"加载的模型训练到第 {checkpoint['epoch']} 轮")
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")

        checkpoint = torch.load(filepath, map_location=device)

        # 验证检查点格式
        required_keys = ['model_state_dict', 'model_config']
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"检查点文件缺少必要字段: {key}")

        config = checkpoint['model_config']
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=config['output_dim'],
            d_state=config.get('d_state', 16),
            d_conv=config.get('d_conv', 4),
            expand=config.get('expand', 2),
            dropout=config.get('dropout', 0.1)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        print(f"模型已从 {filepath} 加载")
        return model, checkpoint

    def getModelInfo(self) -> Dict[str, Any]:
        """
        获取模型配置信息

        返回模型的基本配置参数和参数统计信息

        Returns:
            Dict[str, Any]: 包含模型配置和统计信息的字典

        Example:
            >>> model = CipherMamba(16, 256, 4, 16)
            >>> info = model.getModelInfo()
            >>> print(f"模型参数总数: {info['total_params']}")
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters()
                               if p.requires_grad)

        # 计算模型大小 (MB)
        param_size = sum(p.numel() * p.element_size()
                         for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size()
                          for b in self.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)

        return {
            'model_name': 'CipherMamba',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'd_state': self.d_state,
            'd_conv': self.d_conv,
            'expand': self.expand,
            'dropout': self.dropout_rate,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb
        }
