import torch
import torch.nn as nn
import os
from typing import Dict, Any, Optional


class CipherLSTM(nn.Module):
    """用于密文预测/明文恢复的LSTM模型

    Args:
        input_dim (int): 输入特征维度，通常为64（16位明文/密文）
        hidden_dim (int): LSTM隐藏层维度
        num_layers (int): LSTM层数
        output_dim (int): 输出维度，通常为8（8位目标位）
        dropout (float, optional): LSTM层之间的dropout概率，默认0.2

        Example:
            >>> model = CipherLSTM(16, 128, 2, 16)
            >>> model.saveModel('checkpoint.pth', optimizer,
            ...                 epoch=10, loss=0.1)
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # 输出0/1概率

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len=1, input_dim)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # 取最后一步输出
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)

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
            >>> model = CipherLSTM(16, 128, 2, 16)
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
                'output_dim': self.output_dim
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
    def loadModel(cls, filepath: str, device: str = 'cpu') -> tuple:
        """
        从检查点文件加载模型

        从保存的检查点文件中恢复模型实例和相关信息

        Args:
            filepath (str): 检查点文件路径
            device (str): 设备类型 ('cpu' 或 'cuda')

        Returns:
            tuple: (模型实例, 检查点信息字典)

        Raises:
            FileNotFoundError: 检查点文件不存在时抛出异常
            KeyError: 检查点文件格式错误时抛出异常

        Example:
            >>> model, checkpoint = CipherLSTM.loadModel('checkpoint.pth')
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
            output_dim=config['output_dim']
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
            >>> model = CipherLSTM(16, 128, 2, 16)
            >>> info = model.getModelInfo()
            >>> print(f"模型参数总数: {info['total_params']}")
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters()
                               if p.requires_grad)

        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
