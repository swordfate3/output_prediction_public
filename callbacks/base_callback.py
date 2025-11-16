#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化回调函数类

定义训练过程中回调函数的基础接口
"""

from typing import List, Dict, Any, Optional


class Callback:
    """
    回调函数基类
    
    定义训练过程中回调函数的基本接口。
    回调函数可以通过trainer_instance直接访问训练器的training_results，
    避免重复的参数传递。
    """
    
    def __init__(self):
        """
        初始化回调函数
        """
        self.trainer_instance = None
    
    def set_trainer(self, trainer_instance):
        """
        设置训练器实例的引用
        
        通过此方法，回调函数可以直接访问训练器的training_results
        和其他属性，避免通过logs参数传递大量数据
        
        Args:
            trainer_instance: 训练器实例（如TrainModel）
            
        Example:
            >>> callback.set_trainer(train_model_instance)
            >>> # 现在可以通过self.trainer_instance.training_results访问数据
        """
        self.trainer_instance = trainer_instance
    
    def get_training_results(self) -> Optional[Dict[str, Any]]:
        """
        获取训练结果数据
        
        便捷方法，用于获取训练器的training_results
        
        Returns:
            dict: 训练结果字典，如果训练器未设置则返回None
            
        Example:
            >>> results = self.get_training_results()
            >>> if results:
            ...     val_loss = results['val_loss']
        """
        if (self.trainer_instance and
                hasattr(self.trainer_instance, 'training_results')):
            return self.trainer_instance.training_results
        return None
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """
        训练开始时调用
        
        Args:
            logs (dict, optional): 额外的训练日志信息（可选）
        """
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        每个epoch结束时调用
        
        Args:
            epoch (int): 当前epoch数
            logs (dict, optional): 额外的训练日志信息（可选）
        """
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """
        训练结束时调用
        
        Args:
            logs (dict, optional): 额外的训练日志信息（可选）
        """
        pass


class CallbackManager:
    """
    回调函数管理器
    
    负责管理和调用注册的回调函数，提供统一的回调接口。
    自动为所有回调函数设置训练器实例的引用。
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None, 
                 trainer_instance=None):
        """
        初始化回调管理器

        Args:
            callbacks (List[Callback], optional): 初始回调函数列表。默认为空列表。
            trainer_instance: 训练器实例（如TrainModel），可选
        """
        self.callbacks = callbacks if callbacks is not None else []
        self.trainer_instance = trainer_instance
        # 为初始回调函数设置训练器引用
        if self.trainer_instance:
            for callback in self.callbacks:
                callback.set_trainer(self.trainer_instance)

    def set_trainer(self, trainer_instance):
        """
        设置训练器实例
        
        为管理器和所有已注册的回调函数设置训练器实例引用
        
        Args:
            trainer_instance: 训练器实例（如TrainModel）
        """
        self.trainer_instance = trainer_instance
        # 为所有已注册的回调函数设置训练器引用
        for callback in self.callbacks:
            callback.set_trainer(trainer_instance)

    def add_callback(self, callback: Callback):
        """
        添加一个回调函数到管理器

        Args:
            callback (Callback): 要添加的回调函数实例
        """
        if not isinstance(callback, Callback):
            raise TypeError("回调函数必须继承自Callback基类")
        self.callbacks.append(callback)
        # 如果训练器实例已设置，为新添加的回调函数设置引用
        if self.trainer_instance:
            callback.set_trainer(self.trainer_instance)

    def remove_callback(self, callback: Callback):
        """
        从管理器中移除指定的回调函数

        Args:
            callback (Callback): 要移除的回调函数实例
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def clear_callbacks(self):
        """
        清空所有回调函数
        """
        self.callbacks.clear()

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """
        在训练开始时调用所有注册的回调函数

        Args:
            logs (Dict[str, Any], optional): 训练日志信息
        """
        for callback in self.callbacks:
            try:
                callback.on_train_begin(logs)
            except Exception as e:
                print(f"回调函数 {callback.__class__.__name__} 在训练开始时出错: {e}")

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        在每个epoch结束时调用所有注册的回调函数

        Args:
            epoch (int): 当前epoch数
            logs (Dict[str, Any], optional): 训练日志信息

        Returns:
            Dict[str, Any]: 聚合所有回调返回的结果
        """
        results = {}
        for callback in self.callbacks:
            try:
                callback_result = callback.on_epoch_end(epoch, logs)
                if callback_result is not None:
                    results.update(callback_result)
            except Exception as e:
                print(f"回调函数 {callback.__class__.__name__} 在epoch结束时出错: {e}")
        return results

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """
        在训练结束时调用所有注册的回调函数

        Args:
            logs (Dict[str, Any], optional): 训练日志信息
        """
        for callback in self.callbacks:
            try:
                callback.on_train_end(logs)
            except Exception as e:
                print(f"回调函数 {callback.__class__.__name__} 在训练结束时出错: {e}")

    def get_callback_count(self) -> int:
        """
        获取当前注册的回调函数数量

        Returns:
            int: 回调函数数量
        """
        return len(self.callbacks)

    def get_callback_names(self) -> List[str]:
        """
        获取所有注册回调函数的类名

        Returns:
            List[str]: 回调函数类名列表
        """
        return [callback.__class__.__name__ for callback in self.callbacks]