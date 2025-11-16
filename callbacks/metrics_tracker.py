#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标跟踪回调函数

实现训练过程中的指标收集、存储和分析功能
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from .base_callback import Callback


class MetricsTrackerCallback(Callback):
    """
    指标跟踪回调函数
    
    负责收集、存储和分析训练过程中的各种指标
    """
    
    def __init__(self, 
                 save_dir: str = "./metrics",
                 experiment_name: str = "training",
                 save_frequency: int = 10,
                 track_gradients: bool = False,
                 track_weights: bool = False):
        """
        初始化指标跟踪回调函数
        
        Args:
            save_dir (str): 指标保存目录
            experiment_name (str): 实验名称
            save_frequency (int): 保存频率（每多少个epoch保存一次）
            track_gradients (bool): 是否跟踪梯度信息
            track_weights (bool): 是否跟踪权重信息
        """
        super().__init__()
        
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.save_frequency = save_frequency
        self.track_gradients = track_gradients
        self.track_weights = track_weights
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 指标存储
        self.metrics_history = {
            'experiment_info': {
                'name': experiment_name,
                'start_time': None,
                'end_time': None,
                'total_epochs': 0,
                'stopped_early': False
            },
            'epochs': [],
            'training_metrics': {},
            'validation_metrics': {},
            'learning_rates': [],
            'custom_metrics': {},
            'model_info': {},
            'hyperparameters': {},
            'system_info': {}
        }
        
        # 统计信息
        self.statistics = {
            'best_metrics': {},
            'worst_metrics': {},
            'average_metrics': {},
            'improvement_epochs': [],
            'degradation_epochs': []
        }
        
        # 记录开始时间
        self.metrics_history['experiment_info']['start_time'] = datetime.now().isoformat()
        
        # 记录系统信息
        self.metrics_history['system_info'] = self._get_system_info()
    

    def on_epoch_end(self, epoch, logs=None):
        """
        每个epoch结束时收集指标
        
        Args:
            epoch (int): 当前epoch数
            logs (dict, optional): 额外的训练日志信息（已弃用，保留兼容性）
        """
        # 记录epoch
        self.metrics_history['epochs'].append(epoch)
        
        # 直接从训练器实例获取训练结果
        training_results = self.get_training_results()
        if training_results:
            self._process_metrics(epoch, training_results)
            
            # 记录学习率
            if 'learning_rate' in training_results:
                self.metrics_history['learning_rates'].append(training_results['learning_rate'])
            
            # 处理自定义数据
            if 'custom_data' in training_results:
                self._process_custom_data(epoch, training_results['custom_data'])
        
        # 更新统计信息
        self._update_statistics(epoch, training_results)
        
        # 按频率保存
        if epoch % self.save_frequency == 0:
            self._save_metrics()
    
    def on_train_end(self, logs=None):
        """
        训练结束时保存最终指标
        
        Args:
            logs (dict, optional): 额外的训练日志信息（已弃用，保留兼容性）
        """
        # 更新实验信息
        self.metrics_history['experiment_info']['end_time'] = datetime.now().isoformat()
        self.metrics_history['experiment_info']['total_epochs'] = len(self.metrics_history['epochs'])
        
        # 直接从训练器实例获取最终训练结果
        training_results = self.get_training_results()
        if training_results:
            self.metrics_history['final_metrics'] = training_results.copy()
        
        # 计算最终统计信息
        self._finalize_statistics()
        
        # 保存最终指标
        self._save_metrics(final=True)
        
        # 生成报告
        self._generate_summary_report()
    

    
    def _process_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        处理训练指标
        
        Args:
            epoch (int): 当前epoch
            metrics (Dict[str, float]): 指标字典
        """
        for metric_name, value in metrics.items():
            # 分类存储指标
            if metric_name.startswith('train_'):
                category = 'training_metrics'
                clean_name = metric_name[6:]  # 移除'train_'前缀
            elif metric_name.startswith('val_'):
                category = 'validation_metrics'
                clean_name = metric_name[4:]  # 移除'val_'前缀
            else:
                category = 'custom_metrics'
                clean_name = metric_name
            
            # 初始化指标列表
            if clean_name not in self.metrics_history[category]:
                self.metrics_history[category][clean_name] = []
            
            # 添加指标值
            self.metrics_history[category][clean_name].append({
                'epoch': epoch,
                'value': value
            })
    
    def _process_custom_data(self, epoch: int, custom_data: Dict[str, Any]) -> None:
        """
        处理自定义数据
        
        Args:
            epoch (int): 当前epoch
            custom_data (Dict[str, Any]): 自定义数据
        """
        for key, value in custom_data.items():
            if key not in self.metrics_history['custom_metrics']:
                self.metrics_history['custom_metrics'][key] = []
            
            self.metrics_history['custom_metrics'][key].append({
                'epoch': epoch,
                'value': value
            })
    
    def _update_statistics(self, epoch: int, metrics: Dict[str, float] = None) -> None:
        """
        更新统计信息
        
        Args:
            epoch (int): 当前epoch
            metrics (Dict[str, float]): 指标字典
        """
        if not metrics:
            return
        
        for metric_name, value in metrics.items():
            # 更新最佳和最差值
            if metric_name not in self.statistics['best_metrics']:
                self.statistics['best_metrics'][metric_name] = {'value': value, 'epoch': epoch}
                self.statistics['worst_metrics'][metric_name] = {'value': value, 'epoch': epoch}
            else:
                # 对于损失类指标，越小越好
                if 'loss' in metric_name.lower():
                    if value < self.statistics['best_metrics'][metric_name]['value']:
                        self.statistics['best_metrics'][metric_name] = {'value': value, 'epoch': epoch}
                        self.statistics['improvement_epochs'].append(epoch)
                    if value > self.statistics['worst_metrics'][metric_name]['value']:
                        self.statistics['worst_metrics'][metric_name] = {'value': value, 'epoch': epoch}
                        self.statistics['degradation_epochs'].append(epoch)
                # 对于准确率类指标，越大越好
                elif 'acc' in metric_name.lower() or 'accuracy' in metric_name.lower():
                    if value > self.statistics['best_metrics'][metric_name]['value']:
                        self.statistics['best_metrics'][metric_name] = {'value': value, 'epoch': epoch}
                        self.statistics['improvement_epochs'].append(epoch)
                    if value < self.statistics['worst_metrics'][metric_name]['value']:
                        self.statistics['worst_metrics'][metric_name] = {'value': value, 'epoch': epoch}
                        self.statistics['degradation_epochs'].append(epoch)
    
    def _finalize_statistics(self) -> None:
        """
        计算最终统计信息
        """
        # 计算平均值
        for category in ['training_metrics', 'validation_metrics']:
            for metric_name, values in self.metrics_history[category].items():
                if values:
                    avg_value = sum(item['value'] for item in values) / len(values)
                    self.statistics['average_metrics'][f"{category}_{metric_name}"] = avg_value
        
        # 去重改善和退化的epoch
        self.statistics['improvement_epochs'] = list(set(self.statistics['improvement_epochs']))
        self.statistics['degradation_epochs'] = list(set(self.statistics['degradation_epochs']))
    
    def _get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            Dict[str, Any]: 系统信息
        """
        import platform
        
        try:
            import psutil
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            # psutil未安装时的简化版本
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'timestamp': datetime.now().isoformat(),
                'note': 'psutil not available for detailed system info'
            }
        except Exception:
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'timestamp': datetime.now().isoformat(),
                'note': 'error getting system info'
            }
    
    def _save_metrics(self, final: bool = False) -> None:
        """
        保存指标到文件
        
        Args:
            final (bool): 是否为最终保存
        """
        try:
            # 保存详细指标
            suffix = "_final" if final else ""
            metrics_file = os.path.join(self.save_dir, f"{self.experiment_name}_metrics{suffix}.json")
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, ensure_ascii=False, indent=2, default=str)
            
            # 保存统计信息
            stats_file = os.path.join(self.save_dir, f"{self.experiment_name}_statistics{suffix}.json")
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.statistics, f, ensure_ascii=False, indent=2, default=str)
            
            if final:
                pass
            
        except Exception as e:
            pass
    
    def _generate_summary_report(self) -> None:
        """
        生成训练摘要报告
        """
        try:
            report_file = os.path.join(self.save_dir, f"{self.experiment_name}_summary.txt")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"训练摘要报告 - {self.experiment_name}\n")
                f.write("=" * 50 + "\n\n")
                
                # 实验信息
                exp_info = self.metrics_history['experiment_info']
                f.write(f"实验名称: {exp_info['name']}\n")
                f.write(f"开始时间: {exp_info['start_time']}\n")
                f.write(f"结束时间: {exp_info['end_time']}\n")
                f.write(f"总轮次: {exp_info['total_epochs']}\n")
                f.write(f"早停: {'是' if exp_info['stopped_early'] else '否'}\n\n")
                
                # 最佳指标
                f.write("最佳指标:\n")
                for metric_name, info in self.statistics['best_metrics'].items():
                    f.write(f"  {metric_name}: {info['value']:.6f} (Epoch {info['epoch']})\n")
                f.write("\n")
                
                # 平均指标
                f.write("平均指标:\n")
                for metric_name, value in self.statistics['average_metrics'].items():
                    f.write(f"  {metric_name}: {value:.6f}\n")
                f.write("\n")
                
                # 改善次数
                f.write(f"指标改善次数: {len(self.statistics['improvement_epochs'])}\n")
                f.write(f"指标退化次数: {len(self.statistics['degradation_epochs'])}\n")
            
            pass
            
        except Exception as e:
            pass
    
    def get_metrics_history(self) -> Dict[str, Any]:
        """
        获取指标历史
        
        Returns:
            Dict[str, Any]: 指标历史数据
        """
        return self.metrics_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return self.statistics.copy()
    
    def get_best_metric(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定指标的最佳值
        
        Args:
            metric_name (str): 指标名称
            
        Returns:
            Optional[Dict[str, Any]]: 最佳指标信息
        """
        return self.statistics['best_metrics'].get(metric_name)
    
    def export_to_csv(self, file_path: str = None) -> str:
        """
        导出指标到CSV文件
        
        Args:
            file_path (str): 导出文件路径
            
        Returns:
            str: 导出的文件路径
        """
        if file_path is None:
            file_path = os.path.join(self.save_dir,
                                     f"{self.experiment_name}_metrics.csv")

        try:
            import pandas as pd

            # 准备数据
            data = []
            epochs = self.metrics_history['epochs']

            for i, epoch in enumerate(epochs):
                row = {'epoch': epoch}

                # 添加训练指标
                for metric_name, values in (
                        self.metrics_history['training_metrics'].items()):
                    if i < len(values):
                        row[f'train_{metric_name}'] = values[i]['value']

                # 添加验证指标
                for metric_name, values in (
                        self.metrics_history['validation_metrics'].items()):
                    if i < len(values):
                        row[f'val_{metric_name}'] = values[i]['value']

                # 添加学习率
                if i < len(self.metrics_history['learning_rates']):
                    row['learning_rate'] = (
                        self.metrics_history['learning_rates'][i])

                data.append(row)

            # 创建DataFrame并保存
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)

            self._log(f"指标已导出至CSV: {file_path}", "info", {
                'file_path': file_path,
                'rows': len(data)
            })

            return file_path

        except ImportError:
            self._log("pandas未安装，无法导出CSV", "warning")
            raise
        except Exception as e:
            self._log(f"导出CSV失败: {e}", "error", {'error': str(e)})
            raise

    def __repr__(self) -> str:
        return (f"MetricsTrackerCallback(save_dir='{self.save_dir}', "
                f"experiment_name='{self.experiment_name}', "
                f"save_frequency={self.save_frequency})")