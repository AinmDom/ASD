import os
# 先导入numpy，确保它是第一个被导入的科学计算库
import numpy as np
# 检查numpy版本
print(f"Using NumPy version: {np.__version__}")

# 导入torch相关库
import torch
import torch.nn as nn
import torch.optim as optim
import yaml  # 添加缺失的yaml导入

# 解决matplotlib和pandas版本不兼容问题
import matplotlib
import types
# 检查并添加缺失的函数
if not hasattr(matplotlib.cbook, '_is_pandas_dataframe'):
    def _is_pandas_dataframe(obj):
        """检查对象是否为pandas DataFrame"""
        try:
            import pandas as pd
            return isinstance(obj, pd.DataFrame)
        except ImportError:
            return False
    
    # 将函数添加到matplotlib.cbook模块
    matplotlib.cbook._is_pandas_dataframe = _is_pandas_dataframe

# 解决numpy和sklearn版本不兼容问题
try:
    from numpy.core.numeric import ComplexWarning
except ImportError:
    # 如果导入失败，手动定义这个异常类
    class ComplexWarning(UserWarning):
        """当将复数转换为实数时发出警告"""
        pass
    # 将异常类添加到numpy.core.numeric模块
    import numpy.core.numeric
    numpy.core.numeric.ComplexWarning = ComplexWarning

# 现在导入torch数据加载器
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import seaborn as sns

# 手动实现所有必要的评估指标函数，避免导入sklearn
def accuracy_score(y_true, y_pred):
    """计算准确率"""
    correct = np.sum(np.array(y_true) == np.array(y_pred))
    return correct / len(y_true)

def precision_score(y_true, y_pred, average='binary'):
    """计算精确率"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_score(y_true, y_pred, average='binary'):
    """计算召回率"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred, average='binary'):
    """计算F1分数"""
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def confusion_matrix(y_true, y_pred):
    """计算混淆矩阵"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 只考虑二分类情况
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def roc_curve(y_true, y_pred):
    """简化版ROC曲线计算"""
    # 这个是简化版本，实际应用中可能需要更精确的实现
    thresholds = np.sort(np.unique(y_pred))[::-1]
    fpr_list = [0.0]
    tpr_list = [0.0]
    
    for threshold in thresholds:
        y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds

def roc_auc_score(y_true, y_pred):
    """简化版ROC AUC计算"""
    # 这个是简化版本，实际应用中可能需要更精确的实现
    try:
        from scipy import integrate
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return integrate.trapz(tpr, fpr)
    except ImportError:
        # 如果scipy也无法导入，使用简化的AUC计算
        print("Warning: scipy not available, using simplified AUC calculation")
        # 排序预测值和真实标签
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sorted_indices = np.argsort(y_pred)
        y_true_sorted = y_true[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]
        
        # 计算真阳性和假阳性
        tp = np.sum(y_true)
        fp = len(y_true) - tp
        
        # 简化的AUC计算
        auc = 0.0
        last_tp = 0
        last_fp = 0
        for i in range(len(y_true_sorted)):
            if i > 0 and y_pred_sorted[i] != y_pred_sorted[i-1]:
                # 当阈值变化时计算矩形面积
                auc += (last_tp * (last_fp - fp))
            if y_true_sorted[i] == 1:
                last_tp = tp
                tp -= 1
            else:
                last_fp = fp
                fp -= 1
        auc += (last_tp * last_fp)
        
        # 归一化
        if last_tp * last_fp > 0:
            auc = auc / (last_tp * last_fp)
        
        return auc

# 导入tqdm用于进度显示
try:
    from tqdm import tqdm
except ImportError:
    # 如果tqdm也无法导入，创建一个简单的替代实现
    print("Warning: tqdm not available, using simple progress display")
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None):
            self.iterable = iterable
            self.desc = desc or "Processing"
            self.total = total or (len(iterable) if iterable is not None else 0)
            self.current = 0
            
        def __iter__(self):
            if self.iterable is not None:
                for item in self.iterable:
                    yield item
                    self.update(1)
        
        def update(self, n=1):
            self.current += n
            if self.total > 0:
                percent = (self.current / self.total) * 100
                print(f"\r{self.desc}: {self.current}/{self.total} ({percent:.1f}%)", end="")
        
        def set_postfix(self, **kwargs):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            print()  # 换行

# 导入自定义模型和数据处理器
try:
    from models.anomaly_detection_model import AudioAnomalyDetector
    from utils.data_processor import create_data_loaders
except ImportError as e:
    print(f"Warning: Error importing custom modules: {e}")
    print("This may cause issues later in the code execution")

class ModelTrainer:
    def __init__(self, config_path):
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        try:
            # 尝试检查CUDA可用性并获取GPU名称
            if torch.cuda.is_available() and self.config['training']['device'] == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                print(f"检测到GPU: {gpu_name}")
                # 检查是否有兼容性警告，但仍然尝试使用GPU
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        except Exception as e:
            # 如果CUDA相关操作失败，退回到CPU
            print(f"CUDA初始化失败: {e}，将使用CPU进行训练")
            self.device = torch.device('cpu')
        
        # 打印最终选择的设备
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = AudioAnomalyDetector(self.config).to(self.device)
        
        # 初始化优化器
        self.optimizer = self.model.get_optimizer(stage=1)
        
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
            'optimizer_state_dict': self.optimizer.state_dict(),
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
        
        # 尝试加载优化器状态字典，如果不存在则忽略
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Warning: Optimizer state dict not found in model file, using default optimizer settings")
        
        # 尝试加载最佳验证损失，如果不存在则保持当前值
        if 'best_val_loss' in checkpoint:
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