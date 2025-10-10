import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from tqdm import tqdm

from models.anomaly_detection_model import AudioAnomalyDetector
from utils.data_processor import create_data_loaders

class ModelTrainer:
    def __init__(self, config_path):
        # 加载配置文件
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config['training']['device'] == 'cuda' else 'cpu'
        )
        
        # 初始化模型
        self.model = AudioAnomalyDetector(self.config).to(self.device)
        
        # 创建保存模型的目录
        self.model_dir = self.config['output']['model_save_path']
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 训练历史记录
        self.history = {
            'stage1': {
                'train_loss': [],
                'val_loss': []
            },
            'stage2': {
                'train_loss': [],
                'val_loss': []
            }
        }
        
        # 早停机制参数
        self.patience = self.config['training']['stage1']['patience']
        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
        # 教师模型（用于Mean Teacher）
        self.teacher_model = None
        
        # 加载数据加载器
        self.loaders = None
        self.val_loader = None
        self._load_data()
        
    def _load_data(self):
        """加载数据加载器"""
        self.loaders, self.val_loader = create_data_loaders(self.config, stage=1)
    
    def train_epoch(self, stage=1):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        
        # 根据训练阶段设置不同的优化器
        optimizer = self.model.get_optimizer(stage)
        
        # 准备数据迭代器
        loaders_to_use = []
        if self.loaders['strong'] is not None:
            loaders_to_use.append((self.loaders['strong'], 'strong'))
        if self.loaders['weak'] is not None:
            loaders_to_use.append((self.loaders['weak'], 'weak'))
        if self.loaders['unlabeled'] is not None:
            loaders_to_use.append((self.loaders['unlabeled'], 'unlabeled'))
        
        # 确保有数据可用
        if not loaders_to_use:
            raise ValueError("没有可用的训练数据加载器")
        
        # 第一阶段：只使用强标记和弱标记数据训练CRNN
        if stage == 1:
            with tqdm(total=sum(len(loader) for loader, _ in loaders_to_use), desc=f'Stage 1 Training') as pbar:
                for loader, data_type in loaders_to_use:
                    for batch in loader:
                        if data_type == 'strong':
                            inputs, labels = batch
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                        elif data_type == 'weak':
                            inputs, labels = batch
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            # 对于弱标记数据，我们只需要一个标签用于整个序列
                            labels = labels.unsqueeze(1).repeat(1, inputs.size(1), 1)
                        else:
                            # 第一阶段不使用无标记数据
                            continue
                        
                        # 梯度清零
                        optimizer.zero_grad()
                        
                        # 前向传播
                        outputs = self.model(inputs)
                        
                        # 计算监督损失（二元交叉熵）
                        loss = self.model.compute_supervised_loss(outputs, labels)
                        
                        # 反向传播和优化
                        loss.backward()
                        optimizer.step()
                        
                        # 累计损失
                        total_loss += loss.item()
                        total_batches += 1
                        
                        # 更新进度条
                        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                        pbar.update(1)
        
        # 第二阶段：使用所有数据，包括Mean Teacher和插值一致性训练
        else:
            # 如果还没有教师模型，创建一个
            if self.teacher_model is None:
                self.teacher_model = AudioAnomalyDetector(self.config).to(self.device)
                self.teacher_model.load_state_dict(self.model.state_dict())
                for param in self.teacher_model.parameters():
                    param.requires_grad = False
            
            with tqdm(total=sum(len(loader) for loader, _ in loaders_to_use), desc=f'Stage 2 Training') as pbar:
                for loader, data_type in loaders_to_use:
                    for batch in loader:
                        if data_type == 'strong':
                            inputs, labels = batch
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            # 复制一份用于插值一致性训练
                            inputs2 = inputs.clone().detach()
                        elif data_type == 'weak':
                            inputs, labels = batch
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            # 对于弱标记数据，我们只需要一个标签用于整个序列
                            labels = labels.unsqueeze(1).repeat(1, inputs.size(1), 1)
                            # 复制一份用于插值一致性训练
                            inputs2 = inputs.clone().detach()
                        else:
                            # 无标记数据只有输入
                            inputs = batch.to(self.device)
                            inputs2 = inputs.clone().detach()
                            labels = None
                        
                        # 梯度清零
                        optimizer.zero_grad()
                        
                        # 前向传播（学生模型）
                        student_outputs = self.model(inputs)
                        student_outputs2 = self.model(inputs2)
                        
                        # 教师模型前向传播
                        with torch.no_grad():
                            teacher_outputs = self.teacher_model(inputs)
                        
                        # 计算损失
                        total_batch_loss = 0.0
                        
                        # 如果有标签，计算监督损失
                        if labels is not None:
                            supervised_loss = self.model.compute_supervised_loss(student_outputs, labels)
                            total_batch_loss += supervised_loss
                        
                        # 计算Mean Teacher损失
                        mean_teacher_loss = self.model.compute_mean_teacher_loss(
                            student_outputs, teacher_outputs, labels
                        )
                        total_batch_loss += mean_teacher_loss
                        
                        # 计算插值一致性训练损失
                        consistency_loss = self.model.compute_interpolation_consistency_loss(
                            student_outputs, student_outputs2
                        )
                        total_batch_loss += consistency_loss
                        
                        # 反向传播和优化
                        total_batch_loss.backward()
                        optimizer.step()
                        
                        # 更新教师模型（指数移动平均）
                        with torch.no_grad():
                            alpha = 0.99  # EMA系数
                            for student_param, teacher_param in zip(
                                self.model.parameters(), self.teacher_model.parameters()
                            ):
                                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
                        
                        # 累计损失
                        total_loss += total_batch_loss.item()
                        total_batches += 1
                        
                        # 更新进度条
                        pbar.set_postfix({'loss': f'{total_batch_loss.item():.6f}'})
                        pbar.update(1)
        
        # 计算平均损失
        if total_batches > 0:
            avg_loss = total_loss / total_batches
        else:
            avg_loss = 0.0
        
        return avg_loss
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        
        if self.val_loader is None:
            print("Warning: 没有可用的验证数据加载器")
            return 0.0
        
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc='Validation') as pbar:
                for batch in self.val_loader:
                    # 强标记的验证数据
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    
                    # 计算监督损失（二元交叉熵）
                    loss = self.model.compute_supervised_loss(outputs, labels)
                    
                    # 累计损失
                    total_loss += loss.item()
                    total_batches += 1
                    
                    # 更新进度条
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                    pbar.update(1)
        
        # 计算平均损失
        if total_batches > 0:
            avg_loss = total_loss / total_batches
        else:
            avg_loss = 0.0
        
        return avg_loss
    
    def train(self):
        """执行两阶段训练"""
        print("开始两阶段训练...")
        
        # 第一阶段：冻结ATST-Frame，只训练CRNN网络
        print("\n===== 第一阶段训练 ====")
        self.model.freeze_atst_frame()
        
        # 获取第一阶段的参数
        stage1_epochs = self.config['training']['stage1']['epochs']
        self.patience = self.config['training']['stage1']['patience']
        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
        # 重新加载第一阶段的数据
        self.loaders, self.val_loader = create_data_loaders(self.config, stage=1)
        
        for epoch in range(stage1_epochs):
            # 检查是否需要早停
            if self.early_stop:
                print("第一阶段早停触发。")
                break
            
            # 训练一个epoch
            train_loss = self.train_epoch(stage=1)
            
            # 验证一个epoch
            val_loss = self.validate_epoch()
            
            # 记录损失
            self.history['stage1']['train_loss'].append(train_loss)
            self.history['stage1']['val_loss'].append(val_loss)
            
            # 打印损失信息
            print(f'Epoch {epoch+1}/{stage1_epochs}, ' 
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # 早停机制
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                # 保存第一阶段的最佳模型
                self.save_model(f'{self.model_dir}/stage1_best_model.pth')
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        
        # 保存第一阶段的最终模型
        self.save_model(f'{self.model_dir}/stage1_final_model.pth')
        
        # 绘制第一阶段的损失曲线
        self.plot_loss_curve(stage=1)
        
        # 第二阶段：微调所有参数
        print("\n===== 第二阶段训练 ====")
        self.model.unfreeze_atst_frame()
        
        # 获取第二阶段的参数
        stage2_epochs = self.config['training']['stage2']['epochs']
        self.patience = self.config['training']['stage2']['patience']
        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
        # 重新加载第二阶段的数据
        self.loaders, self.val_loader = create_data_loaders(self.config, stage=2)
        
        for epoch in range(stage2_epochs):
            # 检查是否需要早停
            if self.early_stop:
                print("第二阶段早停触发。")
                break
            
            # 训练一个epoch
            train_loss = self.train_epoch(stage=2)
            
            # 验证一个epoch
            val_loss = self.validate_epoch()
            
            # 记录损失
            self.history['stage2']['train_loss'].append(train_loss)
            self.history['stage2']['val_loss'].append(val_loss)
            
            # 打印损失信息
            print(f'Epoch {epoch+1}/{stage2_epochs}, ' 
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # 早停机制
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                # 保存第二阶段的最佳模型
                self.save_model(f'{self.model_dir}/stage2_best_model.pth')
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        
        # 保存第二阶段的最终模型
        self.save_model(f'{self.model_dir}/stage2_final_model.pth')
        
        # 绘制第二阶段的损失曲线
        self.plot_loss_curve(stage=2)
        
        # 绘制完整的训练历史曲线
        self.plot_complete_training_history()
        
        print("\n两阶段训练完成！")
    
    def save_model(self, file_path=None):
        """保存模型"""
        # 创建模型保存目录
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 如果没有提供文件路径，则使用默认路径
        if file_path is None:
            file_path = f'{self.model_dir}/best_model.pth'
        
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }, file_path)
        
        print(f"模型已保存至 {file_path}")
    
    def load_model(self, model_path=None):
        """加载模型"""
        if model_path is None:
            model_path = os.path.join(self.config['output']['model_save_path'], 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Model loaded from {model_path}")
    
    def plot_loss_curve(self, stage=None):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(10, 6))
        
        if stage is not None:
            # 绘制特定阶段的损失曲线
            train_losses = self.history[f'stage{stage}']['train_loss']
            val_losses = self.history[f'stage{stage}']['val_loss']
            
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title(f'Stage {stage} Training and Validation Loss')
            
            # 保存特定阶段的损失曲线
            filename = f'{self.model_dir}/stage{stage}_loss_curve.png'
        else:
            # 绘制所有阶段的组合损失曲线
            all_train_losses = []
            all_val_losses = []
            
            for s in [1, 2]:
                all_train_losses.extend(self.history[f'stage{s}']['train_loss'])
                all_val_losses.extend(self.history[f'stage{s}']['val_loss'])
            
            plt.plot(all_train_losses, label='Train Loss')
            plt.plot(all_val_losses, label='Validation Loss')
            plt.title('Combined Training and Validation Loss')
            
            # 保存组合损失曲线
            filename = f'{self.model_dir}/combined_loss_curve.png'
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        
        print(f"损失曲线已保存至 {filename}")
    
    def plot_complete_training_history(self):
        """绘制完整的训练历史曲线，包括两个阶段"""
        plt.figure(figsize=(12, 8))
        
        # 合并两个阶段的损失数据
        stage1_train = self.history['stage1']['train_loss']
        stage1_val = self.history['stage1']['val_loss']
        stage2_train = self.history['stage2']['train_loss']
        stage2_val = self.history['stage2']['val_loss']
        
        # 计算所有epoch的数量
        total_epochs = len(stage1_train) + len(stage2_train)
        
        # 绘制训练损失
        plt.plot(range(1, len(stage1_train) + 1), stage1_train, 'b-', label='Stage 1 Train Loss')
        plt.plot(range(len(stage1_train) + 1, total_epochs + 1), stage2_train, 'b--', label='Stage 2 Train Loss')
        
        # 绘制验证损失
        plt.plot(range(1, len(stage1_val) + 1), stage1_val, 'r-', label='Stage 1 Validation Loss')
        plt.plot(range(len(stage1_val) + 1, total_epochs + 1), stage2_val, 'r--', label='Stage 2 Validation Loss')
        
        # 添加阶段分隔线
        plt.axvline(x=len(stage1_train), color='gray', linestyle='--', alpha=0.5)
        plt.text(len(stage1_train) / 2, plt.ylim()[1] * 0.9, 'Stage 1', ha='center')
        plt.text(len(stage1_train) + len(stage2_train) / 2, plt.ylim()[1] * 0.9, 'Stage 2', ha='center')
        
        plt.title('Complete Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.model_dir}/complete_training_history.png')
        plt.close()
        
        print(f"完整训练历史曲线已保存至 {self.model_dir}/complete_training_history.png")
    
    def evaluate(self, test_loader, threshold=None):
        """评估模型性能"""
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc='Evaluating') as pbar:
                for batch in test_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    
                    # 对于二分类任务，使用输出的概率值
                    probabilities = torch.sigmoid(outputs)
                    
                    # 将预测结果和真实标签添加到列表中
                    y_pred.extend(probabilities.cpu().numpy().flatten())
                    y_true.extend(labels.cpu().numpy().flatten())
                    
                    # 更新进度条
                    pbar.update(1)
        
        # 如果没有提供阈值，则使用默认阈值0.5
        if threshold is None:
            threshold = 0.5
        
        # 根据阈值生成二分类预测结果
        binary_predictions = [1 if prob > threshold else 0 for prob in y_pred]
        
        # 计算评估指标
        metrics = self.compute_metrics(y_true, binary_predictions)
        
        return metrics
    
    def compute_metrics(self, y_true, y_pred):
        """计算评估指标"""
        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)
        
        # 计算精确率
        precision = precision_score(y_true, y_pred, average='binary')
        
        # 计算召回率
        recall = recall_score(y_true, y_pred, average='binary')
        
        # 计算F1分数
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        
        # 绘制并保存ROC曲线
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        roc_path = f'{self.model_dir}/roc_curve.png'
        plt.savefig(roc_path)
        plt.close()
        
        print(f"ROC曲线已保存至 {roc_path}")
        
        # 绘制并保存混淆矩阵
        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        cm_path = f'{self.model_dir}/confusion_matrix.png'
        plt.savefig(cm_path)
        plt.close()
        
        print(f"混淆矩阵已保存至 {cm_path}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'confusion_matrix': cm.tolist()
        }

if __name__ == '__main__':
    # 主函数，用于训练模型
    config_path = '../configs/config.yaml'
    trainer = ModelTrainer(config_path)
    trainer.train()