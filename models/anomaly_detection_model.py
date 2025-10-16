import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ATSTFrame(nn.Module):
    """ATST-Frame 模块 用于音频帧级别的特征提取"""
    def __init__(self, input_dim=128, hidden_dim=256, num_heads=4):
        super(ATSTFrame, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 投影层将输入映射到高维空间
        self.proj = nn.Linear(input_dim, hidden_dim)
        
        # 多头自注意力机制
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) -> (seq_len, batch_size, input_dim)
        x = x.permute(1, 0, 2)
        
        # 投影到高维空间
        residual = x
        x = self.proj(x)
        x = self.layer_norm1(x + residual)
        
        # 自注意力机制
        residual = x
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm2(x + attn_output)
        
        # 前馈网络
        residual = x
        x = self.feed_forward(x)
        x = residual + x
        
        # 恢复原始维度顺序
        x = x.permute(1, 0, 2)
        
        return x

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim=1, num_mels=128, cnn_channels=64, dropout_rate=0.3):
        super(CNNFeatureExtractor, self).__init__()
        
        # 卷积层提取特征
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(cnn_channels * 2, cnn_channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        # 计算CNN输出特征维度
        # 假设输入形状为 (batch_size, 1, num_mels, time_steps)
        # 经过三次池化后，num_mels 变为 num_mels / 8
        # 这里使用动态计算，避免硬编码
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        # 注意：实际的全连接层将在AudioAnomalyDetector中动态创建


class AudioAnomalyDetector(nn.Module):
    def __init__(self, config):
        super(AudioAnomalyDetector, self).__init__()
        
        # 从配置中获取参数
        self.input_dim = config['model']['input_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.output_dim = config['model']['output_dim']
        self.num_layers = config['model']['num_layers']
        self.dropout_rate = config['model']['dropout_rate']
        self.num_mels = config['data']['n_mels']
        
        # ATST-Frame 模块
        self.atst_frame = ATSTFrame(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim)
        
        # CNN特征提取器
        self.cnn = CNNFeatureExtractor(
            input_dim=self.input_dim,
            num_mels=self.num_mels,
            cnn_channels=64,
            dropout_rate=self.dropout_rate
        )
        
        # CRNN 网络（CNN + RNN）
        # 计算CNN输出维度
        # 假设输入梅尔频谱图的时间维度为 time_steps
        # 经过三次池化后，时间维度变为 time_steps / 8
        # 由于时间维度是可变的，我们将在第一次前向传播时动态计算
        self.cnn_output_dim = None
        
        # 动态创建全连接层
        self.fc_cnn = None
        
        # LSTM编码器和解码器
        self.encoder = nn.LSTM(
            self.hidden_dim, 
            self.hidden_dim, 
            self.num_layers, 
            batch_first=True, 
            dropout=self.dropout_rate
        )
        self.decoder = nn.LSTM(
            self.hidden_dim, 
            self.hidden_dim, 
            self.num_layers, 
            batch_first=True, 
            dropout=self.dropout_rate
        )
        
        # 输出层（用于异常检测的二分类）
        self.fc_output = nn.Linear(self.hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Dropout层
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # 二元交叉熵损失函数
        self.bce_loss = nn.BCELoss()
        
    def _initialize_fc_cnn(self, x):
        # 动态计算CNN输出维度并初始化全连接层
        with torch.no_grad():
            # 首先通过CNN
            cnn_out = self.cnn.conv1(x)
            cnn_out = self.cnn.conv2(cnn_out)
            cnn_out = self.cnn.conv3(cnn_out)
            cnn_out = self.cnn.flatten(cnn_out)
            
            # 获取CNN输出维度
            cnn_output_dim = cnn_out.shape[1]
            
            # 创建全连接层
            self.fc_cnn = nn.Linear(cnn_output_dim, self.hidden_dim).to(x.device)
            self.cnn_output_dim = cnn_output_dim
    
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, n_mels, frame_time_dim)
        # 例如: (batch_size, 31, 1, 128, 9)
        batch_size, num_frames, channels, n_mels, frame_time_dim = x.shape
        
        # 重塑输入，将帧维度合并到批次维度
        x = x.view(batch_size * num_frames, channels, n_mels, frame_time_dim)
        
        # 如果是第一次前向传播，初始化全连接层
        if self.fc_cnn is None:
            self._initialize_fc_cnn(x)
        
        # CNN特征提取
        cnn_out = self.cnn.conv1(x)
        cnn_out = self.cnn.conv2(cnn_out)
        cnn_out = self.cnn.conv3(cnn_out)
        cnn_out = self.cnn.flatten(cnn_out)
        cnn_out = self.fc_cnn(cnn_out)
        cnn_out = self.cnn.dropout(cnn_out)
        
        # 重塑为序列数据，恢复帧维度
        cnn_out = cnn_out.view(batch_size, num_frames, self.hidden_dim)
        
        # 应用ATST-Frame（如果未冻结）
        if self.atst_frame.training:
            atst_out = self.atst_frame(cnn_out)
        else:
            # 如果ATST-Frame被冻结，直接使用CNN输出
            atst_out = cnn_out
        
        # LSTM编码
        encoder_out, (hidden, cell) = self.encoder(atst_out)
        
        # LSTM解码
        decoder_out, _ = self.decoder(encoder_out, (hidden, cell))
        
        # 输出预测（二分类）
        output = self.fc_output(self.dropout(decoder_out))
        output = torch.squeeze(output, dim=-1)  # 移除最后一个维度，匹配标签形状
        output = self.sigmoid(output)  # 应用sigmoid激活函数
        
        return output
    
    def freeze_atst_frame(self):
        """冻结ATST-Frame模块"""
        for param in self.atst_frame.parameters():
            param.requires_grad = False
        self.atst_frame.eval()
    
    def unfreeze_atst_frame(self):
        """解冻ATST-Frame模块"""
        for param in self.atst_frame.parameters():
            param.requires_grad = True
        self.atst_frame.train()
    
    def compute_supervised_loss(self, predictions, labels):
        """计算监督损失（二元交叉熵）"""
        return self.bce_loss(predictions, labels.float())
    
    def compute_mean_teacher_loss(self, student_predictions, teacher_predictions, labels=None):
        """计算mean teacher损失"""
        # Mean Teacher损失：学生模型预测与教师模型预测之间的一致性
        consistency_loss = F.mse_loss(student_predictions, teacher_predictions.detach())
        
        # 如果提供了标签，还可以加入监督损失
        if labels is not None:
            supervised_loss = self.compute_supervised_loss(student_predictions, labels)
            return supervised_loss + consistency_loss
        
        return consistency_loss
    
    def compute_interpolation_consistency_loss(self, predictions1, predictions2, mixing_coef=0.5):
        """计算插值一致性训练损失"""
        # 计算两个预测之间的插值
        interpolated_preds = mixing_coef * predictions1 + (1 - mixing_coef) * predictions2
        
        # 计算与原始预测的一致性损失
        loss1 = F.mse_loss(predictions1, interpolated_preds.detach())
        loss2 = F.mse_loss(predictions2, interpolated_preds.detach())
        
        return (loss1 + loss2) / 2
    
    def detect_anomalies(self, input_data, threshold=0.5):
        """使用模型进行异常检测"""
        with torch.no_grad():
            predictions = self.forward(input_data)
            # 取所有帧预测的平均值作为最终分数
            anomaly_score = torch.mean(predictions).item()
        
        # 根据阈值判断异常
        is_anomaly = anomaly_score > threshold
        
        return is_anomaly, anomaly_score
    
    def get_optimizer(self, stage=1):
        """根据训练阶段返回优化器"""
        if stage == 1:
            # 第一阶段：只训练CRNN网络
            crnn_params = (list(self.cnn.parameters()) + \
                          list(self.fc_cnn.parameters() if self.fc_cnn is not None else []) +\
                          list(self.encoder.parameters()) + \
                          list(self.decoder.parameters()) + \
                          list(self.fc_output.parameters()))
            optimizer = torch.optim.Adam(crnn_params, lr=1e-3)
        else:
            # 第二阶段：微调所有参数，但CNN和RNN使用不同的学习率
            cnn_params = (list(self.cnn.parameters()) + \
                         list(self.fc_cnn.parameters() if self.fc_cnn is not None else []))
            rnn_params = (list(self.encoder.parameters()) + \
                         list(self.decoder.parameters()) + \
                         list(self.fc_output.parameters()))
            atst_params = list(self.atst_frame.parameters()) if self.atst_frame.training else []
            
            optimizer = torch.optim.Adam([
                {'params': cnn_params, 'lr': 2e-4},
                {'params': rnn_params, 'lr': 2e-3},
                {'params': atst_params, 'lr': 2e-4}
            ])
        
        return optimizer