import os
import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random
from tqdm import tqdm

class AudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=44100, n_mels=128, hop_length=704, 
                 n_fft=5632, duration=5.0, augment=False, 
                 augmentation_prob=0.5, mixup_prob=0.25, freq_warp_prob=0.25, 
                 frame_duration=0.128, frame_hop=0.016, data_type='strong'):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.duration = duration
        self.augment = augment
        self.augmentation_prob = augmentation_prob
        self.mixup_prob = mixup_prob
        self.freq_warp_prob = freq_warp_prob
        self.frame_duration = frame_duration
        self.frame_hop = frame_hop
        self.data_type = data_type  # 'strong', 'weak', 'unlabeled'
        
        # 计算一帧的样本数
        self.frame_samples = int(sample_rate * frame_duration)
        self.hop_samples = int(sample_rate * frame_hop)
        
        self.file_list = self._get_file_list()
        self.max_samples = int(sample_rate * duration)

    def _get_file_list(self):
        file_list = []
        # 尝试从不同的子目录加载不同类型的数据
        if self.data_type == 'strong':
            search_dir = os.path.join(self.data_dir, 'strong')
        elif self.data_type == 'weak':
            search_dir = os.path.join(self.data_dir, 'weak')
        elif self.data_type == 'unlabeled':
            search_dir = os.path.join(self.data_dir, 'unlabeled')
        else:
            search_dir = self.data_dir
        
        # 如果特定类型的目录不存在，则使用整个数据集
        if not os.path.exists(search_dir):
            search_dir = self.data_dir
        
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_list.append(os.path.join(root, file))
        
        # 如果没有找到文件，尝试使用整个数据集目录
        if not file_list:
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith('.wav'):
                        file_list.append(os.path.join(root, file))
        
        return file_list

    def _load_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        # 统一音频长度
        if len(y) > self.max_samples:
            y = y[:self.max_samples]
        elif len(y) < self.max_samples:
            y = np.pad(y, (0, self.max_samples - len(y)), 'constant')
        return y

    def _extract_mel_spectrogram(self, y):
        # 计算梅尔频谱图
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        
        # 转换为对数刻度
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # 频率扭曲增强（如果之前选择了这种增强方式）
        if hasattr(self, 'apply_freq_warp') and self.apply_freq_warp:
            # 使用librosa的相位_vocoder实现简单的频率扭曲
            # 这是一个简化版本，实际可能需要更复杂的实现
            rate = random.uniform(0.9, 1.1)
            mel_spectrogram = librosa.core.phase_vocoder(mel_spectrogram, rate=rate)
            # 调整回原始形状
            if mel_spectrogram.shape[1] > self.n_mels:
                mel_spectrogram = mel_spectrogram[:, :self.n_mels]
            elif mel_spectrogram.shape[1] < self.n_mels:
                mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, self.n_mels - mel_spectrogram.shape[1])), 'constant')
        
        # 归一化
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min() + 1e-8)
        
        # 增加通道维度 (C, H, W) 格式
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        
        return mel_spectrogram

    def _augment(self, y):
        # 根据配置的概率决定是否进行增强
        if random.random() < self.augmentation_prob:
            return y
        
        # 随机选择增强方式
        augment_type = random.choice(['mixup', 'freq_warp'])
        
        if augment_type == 'mixup' and random.random() < self.mixup_prob:
            # 选择另一个随机音频进行mixup
            other_idx = random.randint(0, len(self.file_list) - 1)
            other_file = self.file_list[other_idx]
            other_y = self._load_audio(other_file)
            
            # 计算mixup系数
            alpha = 0.2  # mixup系数
            lam = np.random.beta(alpha, alpha)
            
            # 执行mixup
            y = lam * y + (1 - lam) * other_y
        
        elif augment_type == 'freq_warp' and random.random() < self.freq_warp_prob:
            # 频率扭曲增强
            # 注意：这需要在特征提取后进行，这里先保持音频不变
            pass
        
        # 标准化到[-1, 1]
        y = np.clip(y, -1, 1)
        
        return y

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        y = self._load_audio(file_path)
        
        # 重置频率扭曲标志
        self.apply_freq_warp = False
        
        if self.augment:
            # 保存增强类型，用于特征提取时的频率扭曲
            original_y = y.copy()
            y = self._augment(y)
            # 如果音频没有变化，可能是选择了频率扭曲但在音频域无法实现
            if np.array_equal(y, original_y) and random.random() < self.freq_warp_prob:
                self.apply_freq_warp = True
        
        # 提取梅尔频谱图
        mel_spectrogram = self._extract_mel_spectrogram(y)
        
        # 分割成帧
        frames = self._split_into_frames(mel_spectrogram)
        
        # 转换为张量
        frames_tensor = torch.tensor(frames, dtype=torch.float32)
        
        # 根据数据类型返回不同的标签或元数据
        if self.data_type == 'strong':
            # 强标记数据，假设有详细的帧级标签
            # 这里简化处理，实际应用中应该从文件中加载真实标签
            labels = torch.zeros(frames_tensor.size(0))
            return frames_tensor, labels
        elif self.data_type == 'weak':
            # 弱标记数据，只有片段级标签
            # 这里简化处理，实际应用中应该从文件中加载真实标签
            label = torch.tensor(0)
            return frames_tensor, label
        else:
            # 未标记数据，只有特征
            return frames_tensor
    
    def _split_into_frames(self, mel_spectrogram):
        # 将梅尔频谱图分割成固定长度的帧
        # 计算一帧在时间维度上的大小
        time_dim = mel_spectrogram.shape[2]
        frame_time_size = int(self.frame_duration * self.sample_rate / self.hop_length)
        hop_time_size = int(self.frame_hop * self.sample_rate / self.hop_length)
        
        frames = []
        start_idx = 0
        while start_idx + frame_time_size <= time_dim:
            frame = mel_spectrogram[:, :, start_idx:start_idx + frame_time_size]
            frames.append(frame)
            start_idx += hop_time_size
        
        # 如果没有足够的帧，用最后一帧填充
        if not frames:
            # 创建一个零填充的帧
            frame = np.zeros((1, self.n_mels, frame_time_size))
            frames.append(frame)
        
        return np.array(frames)

def create_data_loaders(config, stage=1):
    data_dir = config['data']['dataset_path']
    sample_rate = config['data']['sample_rate']
    n_mels = config['data']['n_mels']
    hop_length = config['data']['hop_length']
    n_fft = config['data']['n_fft']
    duration = config['data']['duration']
    frame_duration = config.get('data', {}).get('frame_duration', 0.128)
    frame_hop = config.get('data', {}).get('frame_hop', 0.016)
    
    # 获取不同类型数据的批大小
    batch_size_strong = config.get('data', {}).get('batch_size_strong', config['data']['batch_size'])
    batch_size_weak = config.get('data', {}).get('batch_size_weak', config['data']['batch_size'])
    batch_size_unlabeled = config.get('data', {}).get('batch_size_unlabeled', config['data']['batch_size'])
    
    # 数据增强参数
    augmentation_prob = config.get('training', {}).get('augmentation_prob', 0.5)
    mixup_prob = config.get('training', {}).get('mixup_prob', 0.25)
    freq_warp_prob = config.get('training', {}).get('freq_warp_prob', 0.25)
    
    # 根据训练阶段决定是否启用数据增强
    augment = True if stage == 1 else True  # 两阶段都使用数据增强
    
    # 尝试加载三种类型的数据
    loaders = {}
    
    try:
        # 加载强标记数据
        strong_dataset = AudioDataset(
            data_dir=os.path.join(data_dir, 'train'),
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            duration=duration,
            augment=augment,
            augmentation_prob=augmentation_prob,
            mixup_prob=mixup_prob,
            freq_warp_prob=freq_warp_prob,
            frame_duration=frame_duration,
            frame_hop=frame_hop,
            data_type='strong'
        )
        
        loaders['strong'] = DataLoader(
            strong_dataset,
            batch_size=batch_size_strong,
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
    except Exception as e:
        print(f"Warning: Failed to load strong labeled data: {e}")
        loaders['strong'] = None
    
    try:
        # 加载弱标记数据
        weak_dataset = AudioDataset(
            data_dir=os.path.join(data_dir, 'train'),
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            duration=duration,
            augment=augment,
            augmentation_prob=augmentation_prob,
            mixup_prob=mixup_prob,
            freq_warp_prob=freq_warp_prob,
            frame_duration=frame_duration,
            frame_hop=frame_hop,
            data_type='weak'
        )
        
        loaders['weak'] = DataLoader(
            weak_dataset,
            batch_size=batch_size_weak,
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
    except Exception as e:
        print(f"Warning: Failed to load weakly labeled data: {e}")
        loaders['weak'] = None
    
    try:
        # 加载无标记数据
        unlabeled_dataset = AudioDataset(
            data_dir=os.path.join(data_dir, 'train'),
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            duration=duration,
            augment=augment,
            augmentation_prob=augmentation_prob,
            mixup_prob=mixup_prob,
            freq_warp_prob=freq_warp_prob,
            frame_duration=frame_duration,
            frame_hop=frame_hop,
            data_type='unlabeled'
        )
        
        loaders['unlabeled'] = DataLoader(
            unlabeled_dataset,
            batch_size=batch_size_unlabeled,
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
    except Exception as e:
        print(f"Warning: Failed to load unlabeled data: {e}")
        loaders['unlabeled'] = None
    
    # 创建验证数据集和加载器
    try:
        val_dataset = AudioDataset(
            data_dir=os.path.join(data_dir, 'val'),
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            duration=duration,
            augment=False,
            frame_duration=frame_duration,
            frame_hop=frame_hop,
            data_type='strong'  # 验证集使用强标记格式
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size_strong,  # 验证时使用强标记数据的批大小
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
    except Exception as e:
        print(f"Warning: Failed to load validation data: {e}")
        val_loader = None
    
    return loaders, val_loader