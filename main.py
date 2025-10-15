import os
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trainer import ModelTrainer
from src.inference import AnomalyDetector


def main():
    """主函数，提供命令行接口"""
    parser = argparse.ArgumentParser(description='交通音频异常事件检测系统')
    
    # 子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    
    # 评估命令
    evaluate_parser = subparsers.add_parser('evaluate', help='评估模型')
    evaluate_parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    evaluate_parser.add_argument('--model', type=str, default=None, help='训练好的模型路径')
    evaluate_parser.add_argument('--data', type=str, required=True, help='评估数据集路径')
    evaluate_parser.add_argument('--threshold', type=float, default=None, help='异常检测阈值')
    
    # 推理命令
    infer_parser = subparsers.add_parser('infer', help='进行异常检测推理')
    infer_parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    infer_parser.add_argument('--model', type=str, default=None, help='训练好的模型路径')
    infer_parser.add_argument('--input', type=str, required=True, help='输入音频文件或目录路径')
    infer_parser.add_argument('--threshold', type=float, default=None, help='异常检测阈值')
    infer_parser.add_argument('--plot', action='store_true', help='生成可视化结果')
    
    # 参数解析
    args = parser.parse_args()
    
    # 如果没有提供命令，显示帮助信息
    if args.command is None:
        parser.print_help()
        return
    
    # 处理不同的命令
    if args.command == 'train':
        # 训练模型
        print(f"开始训练模型，配置文件: {args.config}")
        trainer = ModelTrainer(args.config)
        trainer.train()
    
    elif args.command == 'evaluate':
        # 评估模型
        print(f"开始评估模型，配置文件: {args.config}")
        # 在实际应用中，这里应该加载测试数据集并进行评估
        # 由于没有提供具体的评估数据集加载逻辑，这里仅作为示例
        print("评估功能尚未完全实现，需要加载带有真实标签的测试数据集")
        
    elif args.command == 'infer':
        # 进行推理
        print(f"开始异常检测推理，配置文件: {args.config}")
        detector = AnomalyDetector(args.config, args.model)
        
        if os.path.isfile(args.input):
            # 处理单个文件
            detector.process_single_file(args.input, args.threshold, args.plot)
        elif os.path.isdir(args.input):
            # 处理目录
            detector.process_directory(args.input, args.threshold, args.plot)
        else:
            print(f"输入路径不存在: {args.input}")


if __name__ == '__main__':
    main()