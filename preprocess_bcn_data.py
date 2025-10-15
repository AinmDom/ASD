import os
import librosa
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

# 读取配置文件
def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理设备配置
    if config.get('training', {}).get('device') == 'auto':
        try:
            import torch
            config['training']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            config['training']['device'] = 'cpu'
            print("警告: 未安装PyTorch，默认使用CPU设备")
    
    return config

# 创建目录结构
def create_directory_structure(base_dir):
    """
    创建数据集目录结构，只包含训练、验证和测试目录
    由于所有数据都是强标注的，不需要区分强/弱标注目录
    """
    # 创建主数据目录
    os.makedirs(base_dir, exist_ok=True)
    
    # 创建训练、验证和测试目录
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    
    # 确保目录存在
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return train_dir, val_dir, test_dir

# 定义正常类和异常类标签
def get_label_categories():
    """
    返回用户定义的正常类和异常类标签
    """
    normal_labels = {'rtn', 'peop', 'brak', 'door', 'busd', 'horn'}
    anomaly_labels = {'rare', 'sire', 'troll', 'whtl', 'blin', 'coug'}
    return normal_labels, anomaly_labels

# 处理音频和标注文件
def process_audio_with_annotations(audio_path, annotation_path, config, output_dir):
    """
    根据标注文件中的精确时间信息裁剪音频样本
    直接从标注中提取起始和结束时间，生成对应的音频片段
    """
    try:
        # 获取标签分类
        normal_labels, anomaly_labels = get_label_categories()
        
        # 读取音频文件
        y, sr = librosa.load(audio_path, sr=config['data']['sample_rate'])
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 读取标注文件
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = f.readlines()
        
        # 解析标注信息
        audio_segments = []
        all_labels = set()
        
        for line in annotations:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 尝试解析不同格式的标注
            try:
                # 优先使用制表符分割，因为用户提供的标注格式是制表符分隔的
                if '\t' in line:
                    parts = line.split('\t')
                else:
                    parts = line.split()
                
                # 确保有足够的部分
                if len(parts) >= 3:
                    # 清理各部分中的空白字符
                    start_time_str = parts[0].strip()
                    end_time_str = parts[1].strip()
                    label = parts[2].strip().lower()
                    
                    # 转换时间戳
                    start_time = float(start_time_str)
                    end_time = float(end_time_str)
                else:
                    continue
                
                all_labels.add(label)
                
                # 计算样本在音频中的起始和结束位置（样本数）
                start_idx = int(start_time * sr)
                end_idx = int(end_time * sr)
                
                # 确保不超出音频范围
                if start_idx >= len(y) or end_idx > len(y):
                    print(f"警告：标注时间超出音频范围，跳过: {line}")
                    continue
                
                # 判断是否为异常
                is_anomaly = label in anomaly_labels
                
                # 保存片段信息
                audio_segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'label': label,
                    'is_anomaly': is_anomaly,
                    'duration': end_time - start_time
                })
                
            except Exception as e:
                print(f"警告：无法解析标注行 '{line}': {e}")
                continue
        
        print(f'解析到的所有标签: {all_labels}')
        print(f'找到 {len(audio_segments)} 个带有标签的音频片段')
        
        # 统计各类标签的数量
        label_counts = {}
        for segment in audio_segments:
            label = segment['label']
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        print(f'各类标签数量统计: {label_counts}')
        
        # 为每个标签类别的数据按照8:1:1划分训练集、验证集和测试集
        import random
        random.seed(42)  # 设置随机种子，保证结果可复现
        
        # 按照标签分组
        label_to_segments = {}
        for idx, segment in enumerate(audio_segments):
            label = segment['label']
            if label not in label_to_segments:
                label_to_segments[label] = []
            label_to_segments[label].append(idx)
        
        # 为每个标签分配数据集类型
        segment_to_split = [None] * len(audio_segments)
        
        for label, segment_indices in label_to_segments.items():
            # 打乱顺序
            random.shuffle(segment_indices)
            total = len(segment_indices)
            train_count = int(total * 0.8)
            val_count = int(total * 0.1)
            
            # 分配数据集
            for i, seg_idx in enumerate(segment_indices):
                if i < train_count:
                    segment_to_split[seg_idx] = 'train'
                elif i < train_count + val_count:
                    segment_to_split[seg_idx] = 'val'
                else:
                    segment_to_split[seg_idx] = 'test'
        
        # 处理每个音频片段
        train_count = 0
        val_count = 0
        test_count = 0
        
        for idx, segment in tqdm(enumerate(audio_segments), desc=f'处理音频片段', total=len(audio_segments)):
            # 确定数据集类型
            split_dir = segment_to_split[idx]
            
            # 更新统计计数
            if split_dir == 'train':
                train_count += 1
            elif split_dir == 'val':
                val_count += 1
            else:  # test
                test_count += 1
            
            # 提取音频片段
            audio_chunk = y[segment['start_idx']:segment['end_idx']]
            
            # 生成文件名（使用时间戳确保唯一性）
            start_time_str = f"{segment['start_time']:.3f}"
            end_time_str = f"{segment['end_time']:.3f}"
            file_name = f"{audio_name}_segment_{idx}_from_{start_time_str}_to_{end_time_str}"
            
            # 保存音频文件
            audio_output_path = os.path.join(output_dir, split_dir, f"{file_name}.wav")
            sf.write(audio_output_path, audio_chunk, sr)
            
            # 创建标签文件
            label_output_path = os.path.join(output_dir, split_dir, f"{file_name}.txt")
            with open(label_output_path, 'w', encoding='utf-8') as f:
                f.write(f"{segment['label']} {1 if segment['is_anomaly'] else 0}")
        
        print(f'数据集分割完成：')
        print(f'- 训练集: {train_count} 个片段')
        print(f'- 验证集: {val_count} 个片段')
        print(f'- 测试集: {test_count} 个片段')
        
        # 计算平均片段时长
        if audio_segments:
            avg_duration = sum(seg['duration'] for seg in audio_segments) / len(audio_segments)
            print(f'平均片段时长: {avg_duration:.3f} 秒')
        
    except Exception as e:
        print(f'处理音频和标注文件失败: {e}')
        import traceback
        traceback.print_exc()

# 主函数
def main():
    try:
        # 加载配置
        config = load_config()
        
        # 获取数据集路径
        dataset_path = config['data']['dataset_path']
        if dataset_path.startswith('./'):
            dataset_path = os.path.join(os.getcwd(), dataset_path[2:])
        
        print(f"\n===== BCN数据集预处理工具 =====")
        print(f"处理后数据将保存到: {dataset_path}")
        
        # 创建目录结构
        train_dir, val_dir, test_dir = create_directory_structure(dataset_path)
        print(f"已创建数据集目录结构")
        
        # 检查是否存在temp目录，如果存在则自动使用
        default_temp_dir = os.path.join(os.getcwd(), 'temp')
        if os.path.exists(default_temp_dir) and os.path.isdir(default_temp_dir):
            print(f"\n检测到默认的'temp'目录: {default_temp_dir}")
            print("按Enter键使用该目录，或输入其他目录路径：")
            user_input = input(f"临时目录路径 [{default_temp_dir}]: ")
            temp_data_dir = user_input.strip() if user_input.strip() else default_temp_dir
        else:
            # 提示用户输入BCN Dataset文件路径
            print("\n请将BCN Dataset的音频文件和对应的标注文件放入临时目录中，")
            print("然后输入该临时目录的路径：")
            temp_data_dir = input("临时目录路径: ")
        
        # 验证临时目录是否存在
        if not os.path.exists(temp_data_dir):
            print(f"错误: 临时目录 '{temp_data_dir}' 不存在")
            return
        
        # 查找临时目录中的音频和标注文件
        audio_files = [f for f in os.listdir(temp_data_dir) if f.endswith('.wav')]
        annotation_files = [f for f in os.listdir(temp_data_dir) if f.endswith(('.txt', '.csv'))]
        
        # 显示找到的文件信息
        print(f'找到 {len(audio_files)} 个音频文件和 {len(annotation_files)} 个标注文件')
        
        # 验证文件数量匹配情况
        if len(audio_files) > 0 and len(annotation_files) > 0:
            print(f"\n找到的音频文件:")
            for i, audio_file in enumerate(audio_files, 1):
                print(f"  {i}. {audio_file}")
            
            print(f"\n找到的标注文件:")
            for i, anno_file in enumerate(annotation_files, 1):
                print(f"  {i}. {anno_file}")
        elif len(audio_files) == 0:
            print("错误: 未找到任何音频文件 (.wav)")
            return
        
        # 处理音频和标注文件
        for audio_file in audio_files:
            audio_path = os.path.join(temp_data_dir, audio_file)
            audio_name = os.path.splitext(audio_file)[0]
            
            print(f'处理音频文件: {audio_file}')
            
            # 优化：精确匹配以"_labels.txt"结尾的标注文件
            found_annotation = False
            expected_anno_file = f"{audio_name}_labels.txt"
            
            if expected_anno_file in annotation_files:
                anno_path = os.path.join(temp_data_dir, expected_anno_file)
                print(f'找到对应的标注文件: {expected_anno_file}')
                # 根据标注文件处理音频
                process_audio_with_annotations(audio_path, anno_path, config, dataset_path)
                found_annotation = True
            else:
                # 备选方案：使用包含关系匹配
                for anno_file in annotation_files:
                    if audio_name in anno_file:
                        anno_path = os.path.join(temp_data_dir, anno_file)
                        print(f'找到对应的标注文件: {anno_file}')
                        # 根据标注文件处理音频
                        process_audio_with_annotations(audio_path, anno_path, config, dataset_path)
                        found_annotation = True
                        break
            
            if not found_annotation:
                print(f'未找到与 {audio_file} 对应的标注文件')
        
        print("\n===== 数据预处理完成！=====")
        print(f"处理后的数据已保存到: {dataset_path}")
        print("目录结构:")
        print(f"- {dataset_path}/train/  # 训练集数据")
        print(f"- {dataset_path}/val/    # 验证集数据")
        print(f"- {dataset_path}/test/   # 测试集数据")
        print("\n标签说明:")
        normal_labels, anomaly_labels = get_label_categories()
        print(f"正常类标签: {', '.join(normal_labels)}")
        print(f"异常类标签: {', '.join(anomaly_labels)}")
        print("\n每个标签文件包含两部分信息: <具体标签> <是否异常(0/1)>")
        print("\n注意: 所有音频片段都是根据标注文件中的精确时间信息裁剪的")
        
    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n程序已终止，请修复错误后重试。")

if __name__ == '__main__':
    main()