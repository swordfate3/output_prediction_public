import torch
import torch.nn as nn
import os
from typing import Dict, Any, Optional


class CipherITransformer(nn.Module):
    """
    用于密文预测/明文恢复的简化 Transformer 模型

    该实现使用 PyTorch 内置的 `TransformerEncoder`，并对齐项目中其他模型的
    通用接口与风格（如 `CipherLSTM`、`CipherMamba`、`CipherResNet`）：
    - 构造参数以输入维度、隐藏维度、层数、输出维度为主；
    - 前向接口为 `forward(x: Tensor) -> Tensor`，返回 0/1 概率；
    - 支持保存/加载模型及查询模型信息；
    - 输入按位数组，常见形状 `(batch, 1, input_dim)` 或 `(batch, input_dim)`。

    Args:
        input_dim (int): 输入特征维度，通常为16或128（按比特位数）
        hidden_dim (int): Transformer 的特征维度（d_model）
        num_layers (int): Transformer 编码层数
        num_heads (int): 多头注意力头数
        output_dim (int): 输出维度，通常为目标比特数（如16）
        dropout (float): Dropout 概率

    Example:
        >>> model = CipherITransformer(input_dim=16, hidden_dim=256,
        ...                            num_layers=4, num_heads=4,
        ...                            output_dim=16)
        >>> x = torch.randint(0, 2, (8, 1, 16)).float()
        >>> y = model(x)  # (8, 16)，每位的预测概率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        output_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.dropout_rate = dropout

        # [ADD] 输入嵌入：将每个比特视为一个 token（长度=input_dim，通道=1）
        self.input_proj = nn.Linear(1, hidden_dim)

        # [ADD] 位置编码（可学习），长度固定为 input_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim, hidden_dim))

        # [ADD] Transformer 编码器（batch_first=True 以简化维度处理）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # [ADD] 输出头：池化 + 线性 + Sigmoid（输出0/1概率）
        self.norm_f = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        将输入按特征维度视作序列长度，对每个比特进行嵌入与注意力编码，
        然后进行池化与线性映射，输出每个目标位的预测概率。

        Args:
            x (torch.Tensor): 输入张量，形状为 `(batch, 1, input_dim)` 或 `(batch, input_dim)`

        Returns:
            torch.Tensor: 输出张量，形状为 `(batch, output_dim)`，取值范围 [0, 1]

        Example:
            >>> x = torch.randint(0, 2, (8, 1, 16)).float()
            >>> y = model(x)
            >>> y.shape
            torch.Size([8, 16])
        """
        # 标准化输入形状为 (batch, input_dim, 1)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # (batch, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, input_dim, 1)

        # 输入嵌入 + 位置编码
        h = self.input_proj(x)  # (batch, input_dim, hidden_dim)
        h = h + self.pos_embed  # 广播到 batch 维度

        # Transformer 编码
        h = self.encoder(h)  # (batch, input_dim, hidden_dim)

        # 简单池化（平均池化 across 序列长度=input_dim）
        h = h.mean(dim=1)  # (batch, hidden_dim)
        h = self.norm_f(h)
        h = self.dropout(h)

        out = self.head(h)  # (batch, output_dim)
        return self.sigmoid(out)

    def saveModel(
        self,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        loss: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        保存模型检查点

        保存模型状态字典、优化器状态、训练元数据等信息到指定文件。

        Args:
            filepath (str): 保存文件路径
            optimizer (torch.optim.Optimizer, optional): 优化器实例
            epoch (int): 当前训练轮数
            loss (float): 当前损失值
            metadata (Dict[str, Any], optional): 额外的元数据信息

        Raises:
            OSError: 文件保存失败时抛出异常

        Example:
            >>> model.saveModel('checkpoint.pth', optimizer, epoch=10, loss=0.1)
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'output_dim': self.output_dim,
                'dropout': self.dropout_rate,
            },
            'epoch': epoch,
            'loss': loss,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if metadata is not None:
            checkpoint['metadata'] = metadata
        torch.save(checkpoint, filepath)

    @classmethod
    def loadModel(cls, filepath: str, device: str = 'cpu') -> tuple:
        """
        从检查点文件加载模型

        从保存的检查点文件中恢复模型实例和相关信息。

        Args:
            filepath (str): 检查点文件路径
            device (str): 设备类型 ('cpu' 或 'cuda')

        Returns:
            tuple: (模型实例, 检查点信息字典)

        Raises:
            FileNotFoundError: 检查点文件不存在时抛出异常
            KeyError: 检查点文件格式错误时抛出异常

        Example:
            >>> model, ckpt = CipherITransformer.loadModel('checkpoint.pth')
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        required = ['model_state_dict', 'model_config']
        for k in required:
            if k not in checkpoint:
                raise KeyError(f"检查点文件缺少必要字段: {k}")
        cfg = checkpoint['model_config']
        model = cls(
            input_dim=cfg['input_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            num_heads=cfg.get('num_heads', 4),
            output_dim=cfg['output_dim'],
            dropout=cfg.get('dropout', 0.1),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model, checkpoint

    def getModelInfo(self) -> Dict[str, Any]:
        """
        获取模型配置信息

        返回模型的基本配置参数和参数统计信息。

        Returns:
            Dict[str, Any]: 包含模型配置和统计信息的字典

        Example:
            >>> info = model.getModelInfo()
            >>> print(info['total_params'])
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'output_dim': self.output_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }
