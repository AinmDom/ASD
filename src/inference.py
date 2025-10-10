import os
import yaml
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.anomaly_detection_model import AudioAnomalyDetector
from utils.data_processor import AudioDataset

class AnomalyDetector:
    def __init__(self, config_path, model_path=None):
        # 加载配置文件
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config['training']['device'] == 'cuda' else 'cpu'
        )
        
        # 初始化模型
        self.model = AudioAnomalyDetector(self.config).to(self.device)
        
        # 加载训练好的模型
        if model_path is None:
            model_path = os.path.join(self.config['output']['model_save_path'], 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")
    
    def preprocess_audio(self, audio_path):
        """预处理单个音频文件"""
        sample_rate = self.config['data']['sample_rate']
        n_mels = self.config['data']['n_mels']
        hop_length = self.config['data']['hop_length']
        n_fft = self.config['data']['n_fft']
        duration = self.config['data']['duration']
        
        # 加载音频文件
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
        # 统一音频长度
        max_samples = int(sample_rate * duration)
        if len(y) > max_samples:
            y = y[:max_samples]
        elif len(y) < max_samples:
            y = np.pad(y, (0, max_samples - len(y)), 'constant')
        
        # 提取梅尔频谱图
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels
        )
        
        # 转换为对数刻度
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # 归一化
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min() + 1e-8)
        
        # 增加通道维度并转换为张量
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        mel_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0)
        
        return mel_tensor, y, sr
    
    def detect_anomaly(self, audio_tensor, threshold=None):
        """检测异常"""
        if threshold is None:
            threshold = self.config['training']['threshold']
        
        with torch.no_grad():
            # 将数据移至设备
            audio_tensor = audio_tensor.to(self.device)
            
            # 检测异常
            is_anomaly, error = self.model.detect_anomalies(audio_tensor, threshold)
        
        return is_anomaly.item(), error.item()
    
    def process_single_file(self, audio_path, threshold=None, plot=False):
        """处理单个音频文件并检测异常"""
        # 预处理音频
        audio_tensor, y, sr = self.preprocess_audio(audio_path)
        
        # 检测异常
        is_anomaly, error = self.detect_anomaly(audio_tensor, threshold)
        
        # 打印结果
        print(f"File: {audio_path}")
        print(f"Anomaly detected: {'Yes' if is_anomaly else 'No'}")
        print(f"Reconstruction error: {error:.6f}")
        
        # 如果需要可视化
        if plot:
            self.visualize_results(audio_tensor, y, sr, is_anomaly, error, os.path.basename(audio_path))
        
        return {
            'file_path': audio_path,
            'is_anomaly': is_anomaly,
            'error': error
        }
    
    def process_directory(self, directory_path, threshold=None, plot=False):
        """处理目录中的所有音频文件"""
        results = []
        
        # 获取目录中的所有音频文件
        audio_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        
        # 处理每个音频文件
        progress_bar = tqdm(audio_files, desc='Processing files')
        for audio_file in progress_bar:
            result = self.process_single_file(audio_file, threshold, plot)
            results.append(result)
            
            # 更新进度条
            anomaly_count = sum(1 for r in results if r['is_anomaly'])
            progress_bar.set_postfix({
                'total_files': len(results),
                'anomalies': anomaly_count
            })
        
        return results
    
    def visualize_results(self, audio_tensor, audio_data, sample_rate, is_anomaly, error, filename):
        """可视化音频和检测结果"""
        # 创建图形
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制波形图
        time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        axs[0].plot(time, audio_data)
        axs[0].set_title(f'Audio Waveform - {filename}')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True)
        
        # 绘制梅尔频谱图
        mel_spectrogram = audio_tensor.squeeze().cpu().numpy()
        librosa.display.specshow(mel_spectrogram, sr=sample_rate, hop_length=self.config['data']['hop_length'],
                               x_axis='time', y_axis='mel', ax=axs[1])
        axs[1].set_title(f'Mel Spectrogram - {filename}')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Mel Frequency')
        fig.colorbar(axs[1].collections[0], ax=axs[1], format='%+2.0f dB')
        
        # 添加异常检测结果
        fig.suptitle(f'Anomaly Detection Result: {"ANOMALY" if is_anomaly else "NORMAL"} (Error: {error:.6f})',
                    fontsize=16, fontweight='bold', color='red' if is_anomaly else 'green')
        
        plt.tight_layout()
        
        # 保存图像
        plot_dir = self.config['output']['plot_dir']
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'{os.path.splitext(filename)[0]}_detection.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Visualization saved to {plot_path}")

if __name__ == '__main__':
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Audio Anomaly Detection')
    parser.add_argument('--config', type=str, default='../configs/config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to audio file or directory')
    parser.add_argument('--threshold', type=float, default=None, help='Anomaly detection threshold')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # 初始化异常检测器
    detector = AnomalyDetector(args.config, args.model)
    
    # 处理输入
    if os.path.isfile(args.input):
        # 处理单个文件
        detector.process_single_file(args.input, args.threshold, args.plot)
    elif os.path.isdir(args.input):
        # 处理目录
        detector.process_directory(args.input, args.threshold, args.plot)
    else:
        print(f"Input path does not exist: {args.input}")