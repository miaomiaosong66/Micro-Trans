#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/01/09 07:21
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : main.py
# @Description   : 主程序入口文件

from pathlib import Path
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Normalizer
from ml_models import train_all_models
import tensorflow as tf
from ml_models import train_evaluate_cnn


def setup_directories():
    """创建必要的目录结构"""
    dirs = ['outputs', 'outputs/models', 'outputs/plots', 'outputs/results']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    """主程序入口"""
    try:

        # 创建目录
        setup_directories()

        # 读取数据
        print("Reading data...")
        df = pd.read_csv('data/Merged_Combined.csv')

        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        print("Preprocessing data...")
        
        # 提取特征和标签
        X = df.iloc[:, :-3]
        y = df.iloc[:, -1]  # AD/PD/ASD/Control

        # 保存特征名称
        feature_names = X.columns.tolist()

        # 行归一化
        normalizer = Normalizer(norm='l1')
        X_normalized = normalizer.fit_transform(X)

        # 标签编码
        labelencoder = LabelEncoder()
        y_encoded = labelencoder.fit_transform(y)
        
        # 打印数据集基本信息
        print(f"Preprocessed {X.shape[0]} samples with {X.shape[1]} features")
        
        # 打印疾病类别的分布
        print("\nDisease Distribution:")
        print("-" * 30)
        disease_counts = pd.Series(y).value_counts()
        disease_percentages = pd.Series(y).value_counts(normalize=True) * 100
        for class_name, count in disease_counts.items():
            percentage = disease_percentages[class_name]
            print(f"{class_name}: {count} samples ({percentage:.2f}%)")

        print("\nClass Labels:")
        print("-" * 30)
        print(f"Disease classes: {labelencoder.classes_}")

        # 获取类别标签列表
        class_labels = list(labelencoder.classes_)
        print("\nDisease classes:", class_labels)
        
        # 训练和评估模型
        print("\nStarting model training and evaluation...")
        results = train_all_models(X_normalized, y_encoded, class_labels)

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()