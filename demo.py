import os
import numpy as np
import soundfile as sf
import librosa
import torch
import random
from tqdm import tqdm

# 生成模拟交通音频数据
def generate_synthetic_data(output_dir, num_normal=100, num_anomalous=20, sample_rate=44100, duration=5.0):
    """生成模拟的交通音频数据用于演示"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'anomalous'), exist_ok=True)
    
    # 生成正常交通音频
    print("生成正常交通音频数据...")
    for i in tqdm(range(num_normal)):
        # 创建基础噪音
        audio = np.random.normal(0, 0.02, int(sample_rate * duration))
        
        # 添加一些汽车引擎声（低频）
        for _ in range(random.randint(1, 3)):
            start_time = random.uniform(0, duration - 2)
            end_time = start_time + random.uniform(0.5, 2)
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            
            # 生成低频正弦波
            t = np.linspace(0, end_time - start_time, end_idx - start_idx)
            freq = random.uniform(50, 200)
            amplitude = random.uniform(0.05, 0.15)
            engine_sound = amplitude * np.sin(2 * np.pi * freq * t)
            
            # 添加到音频中
            audio[start_idx:end_idx] += engine_sound
        
        # 限制幅度
        audio = np.clip(audio, -1, 1)
        
        # 保存音频
        output_path = os.path.join(output_dir, 'normal', f'normal_{i}.wav')
        sf.write(output_path, audio, sample_rate)
    
    # 生成异常交通音频
    print("生成异常交通音频数据...")
    for i in tqdm(range(num_anomalous)):
        # 创建基础噪音
        audio = np.random.normal(0, 0.02, int(sample_rate * duration))
        
        # 添加一些正常声音
        for _ in range(random.randint(1, 3)):
            start_time = random.uniform(0, duration - 2)
            end_time = start_time + random.uniform(0.5, 2)
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            
            # 生成低频正弦波
            t = np.linspace(0, end_time - start_time, end_idx - start_idx)
            freq = random.uniform(50, 200)
            amplitude = random.uniform(0.05, 0.15)
            engine_sound = amplitude * np.sin(2 * np.pi * freq * t)
            
            # 添加到音频中
            audio[start_idx:end_idx] += engine_sound
        
        # 添加异常声音（如碰撞声、急刹车声等）
        anomaly_type = random.choice(['collision', 'braking', 'honking'])
        
        if anomaly_type == 'collision':
            # 碰撞声（短而强的噪声）
            start_time = random.uniform(1, duration - 1)
            start_idx = int(start_time * sample_rate)
            
            # 创建碰撞声
            collision_duration = 0.5
            collision_samples = int(collision_duration * sample_rate)
            collision_sound = np.random.normal(0, 0.5, collision_samples)
            collision_sound = collision_sound * np.exp(-np.linspace(0, 5, collision_samples))
            
            # 添加到音频中
            end_idx = min(start_idx + collision_samples, len(audio))
            audio[start_idx:end_idx] += collision_sound[:end_idx - start_idx]
        
        elif anomaly_type == 'braking':
            # 急刹车声（高频噪声）
            start_time = random.uniform(1, duration - 2)
            end_time = start_time + 1.5
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            
            # 创建刹车声
            t = np.linspace(0, end_time - start_time, end_idx - start_idx)
            freq = 1000 + 500 * np.exp(-t)
            amplitude = 0.2 * np.exp(-t * 0.5)
            braking_sound = amplitude * np.sin(2 * np.pi * freq * t)
            
            # 添加到音频中
            audio[start_idx:end_idx] += braking_sound
        
        elif anomaly_type == 'honking':
            # 鸣笛声（高频正弦波）
            start_time = random.uniform(1, duration - 1)
            end_time = start_time + random.uniform(0.5, 1.5)
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            
            # 创建鸣笛声
            t = np.linspace(0, end_time - start_time, end_idx - start_idx)
            freq = 1000 + 500 * np.sin(2 * np.pi * 2 * t)  # 变化的频率
            amplitude = 0.2
            honking_sound = amplitude * np.sin(2 * np.pi * freq * t)
            
            # 添加到音频中
            audio[start_idx:end_idx] += honking_sound
        
        # 限制幅度
        audio = np.clip(audio, -1, 1)
        
        # 保存音频
        output_path = os.path.join(output_dir, 'anomalous', f'anomalous_{i}.wav')
        sf.write(output_path, audio, sample_rate)
    
    print(f"模拟数据生成完成，保存在 {output_dir} 目录下")
    print(f"正常音频: {num_normal} 个文件")
    print(f"异常音频: {num_anomalous} 个文件")


def demo_pipeline():
    """演示完整的异常检测流程"""
    print("=== 交通音频异常事件检测系统演示 ===")
    
    # 1. 生成模拟数据
    data_dir = './data/synthetic_data'
    generate_synthetic_data(data_dir)
    
    # 2. 修改配置文件以使用生成的数据
    config_path = './configs/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    # 更新配置文件中的数据集路径
    updated_config = config_content.replace('./data/BCNDataset', data_dir)
    
    # 保存临时配置文件
    temp_config_path = './configs/temp_config.yaml'
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        f.write(updated_config)
    
    print(f"配置文件已更新，使用临时配置: {temp_config_path}")
    
    # 3. 提示用户运行训练和推理
    print("\n--- 下一步操作指南 ---")
    print("1. 训练模型：")
    print(f"   python main.py train --config {temp_config_path}")
    print("\n2. 运行异常检测：")
    print(f"   python main.py infer --config {temp_config_path} --input {data_dir} --plot")
    print("\n3. 评估模型性能（需要修改代码添加真实标签）：")
    print(f"   python main.py evaluate --config {temp_config_path} --data {data_dir}")
    print("\n注意：由于这是演示数据，训练效果可能不如真实数据集。")


if __name__ == '__main__':
    demo_pipeline()