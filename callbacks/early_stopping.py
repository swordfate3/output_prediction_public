#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
早期停止回调函数

实现训练过程中的早期停止功能，监控指定指标并在满足条件时停止训练
"""

import numpy as np
from typing import Dict, Any, Optional
from .base_callback import Callback


class EarlyStoppingCallback(Callback):
    """
    早期停止回调函数
    
    监控指定的指标，当指标在一定轮次内没有改善时停止训练
    """
    
    def __init__(self, 
                 monitor: str = 'val_acc',
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'max',
                 restore_best_weights: bool = True):
        """
        初始化早期停止回调函数
        
        Args:
            monitor (str): 要监控的指标名称
            patience (int): 等待改善的轮次数
            min_delta (float): 认为是改善的最小变化量
            mode (str): 'min' 表示指标越小越好，'max' 表示指标越大越好
            restore_best_weights (bool): 是否在停止时恢复最佳权重

        """
        super().__init__()
        
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        # 根据模式设置比较函数和初始值
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
            self.best_value = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
            self.best_value = -np.inf
        else:
            raise ValueError(f"模式 '{mode}' 不支持，请使用 'min' 或 'max'")
        
        # 内部状态
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_value = np.inf if self.mode == 'min' else -np.inf
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        每个epoch结束时检查是否需要早停
        
        Args:
            epoch (int): 当前epoch数
            logs (dict, optional): 额外的训练日志信息（可选）
            
        Returns:
            dict: 包含早停信息的字典
        """
        result = {'early_stop': False, 'best_epoch': self.best_epoch}
        
        # 直接从训练器实例获取训练结果
        training_results = self.get_training_results()
        if training_results is None:
            print("警告: 无法获取训练结果数据")
            return result
        
        # 从training_results中获取监控指标
        current_value = training_results.get(self.monitor)
        if current_value is None:
            print(f"警告: 监控指标 '{self.monitor}' 未在训练结果中找到")
            return result
        
        # 检查是否有改善
        if self.monitor_op(current_value - self.min_delta, self.best_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            print(f"Epoch {epoch}: {self.monitor} 改善到 {current_value:.6f}")
        else:
            self.wait += 1
            print(f"Epoch {epoch}: {self.monitor} 没有改善 ({current_value:.6f}), 等待 {self.wait}/{self.patience}")
        
        # 检查是否需要早停
        if self.wait >= self.patience:
            print(f"\n早停触发! 在epoch {epoch}停止训练")
            print(f"最佳{self.monitor}: {self.best_value:.6f} (epoch {self.best_epoch})")
            result['early_stop'] = True
            
            # 直接更新训练器的training_results
            if (self.trainer_instance and
                hasattr(self.trainer_instance, 'training_results')):
                self.trainer_instance.training_results['early_stopped'] = True
                self.trainer_instance.training_results['best_epoch'] = self.best_epoch
                self.trainer_instance.training_results['best_' + self.monitor] = self.best_value
        
        return result
    
    def on_train_end(self, logs=None):
        """
        训练结束时的处理
        """
        pass
    
    def get_best_value(self) -> float:
        """
        获取最佳指标值
        
        Returns:
            float: 最佳指标值
        """
        return self.best_value
    
    def get_best_epoch(self) -> int:
        """
        获取最佳轮次
        
        Returns:
            int: 最佳轮次
        """
        return self.best_epoch
    
    def get_best_weights(self):
        """
        获取最佳权重
        
        Returns:
            最佳模型权重
        """
        return self.best_weights
    
    def reset(self) -> None:
        """
        重置早停状态
        """
        self.best_value = np.Inf if self.mode == 'min' else -np.Inf
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        pass
    
    def __repr__(self) -> str:
        return (f"EarlyStoppingCallback(monitor='{self.monitor}', patience={self.patience}, "
                f"mode='{self.mode}', min_delta={self.min_delta})")