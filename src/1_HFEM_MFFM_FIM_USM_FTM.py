#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/01/17 17:04
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : 1_HFEM_MFFM_FIM_USM_FTM.py
# @Description   : 



import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 添加在代码最开始
import sys
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, Normalizer
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                           precision_score, recall_score, confusion_matrix)

def set_global_random_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # Set environment variables for even more deterministic behavior
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_directories():
    """Create necessary directory structure"""
    dirs = ['outputs/results']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)




class TaxonomyLoader:
    """Load and process bacterial taxonomy information from CSV"""
    
    @staticmethod
    def load_taxonomy(csv_path):
        """Load taxonomy information from CSV file
        
        Args:
            csv_path: Path to the taxonomy CSV file
            
        Returns:
            DataFrame containing processed taxonomy information
        """
        try:
            # 显式指定 UTF-8 编码读取CSV
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # 删除可能存在的无名列
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)
                
            return df
            
        except UnicodeDecodeError:
            print("Error: Please ensure the CSV file is saved with UTF-8 encoding")
            raise
        except FileNotFoundError:
            print(f"Error: Could not find taxonomy file at {csv_path}")
            raise
        except Exception as e:
            print(f"Unexpected error while loading taxonomy: {str(e)}")
            raise
    
    @staticmethod
    def extract_taxa_by_level(taxonomy_df):
        """Extract taxa lists for each taxonomic level
        
        Args:
            taxonomy_df: Loaded taxonomy DataFrame
            
        Returns:
            Dictionary containing taxa lists for each level
        """
        taxa_by_level = {
            'phylum': set(taxonomy_df['Phylum'].dropna().unique()),
            'class': set(taxonomy_df['Class'].dropna().unique()),
            'order': set(taxonomy_df['Order'].dropna().unique()),
            'family': set(taxonomy_df['Family'].dropna().unique()),
            'genus': set(taxonomy_df['Genus'].dropna().unique())
        }
        
        # Convert sets to sorted lists
        return {level: sorted(list(taxa)) for level, taxa in taxa_by_level.items()}
    
    @staticmethod
    def get_feature_mapping(taxonomy_df):
        """Create mapping between features and their taxonomic levels
        
        Args:
            taxonomy_df: Loaded taxonomy DataFrame
            
        Returns:
            Dictionary mapping feature names to their taxonomic levels
        """
        feature_mapping = {}
        
        # 处理每个分类层级
        for idx, row in taxonomy_df.iterrows():
            # 处理属级
            if pd.notna(row['Genus']):
                feature_mapping[row['Genus']] = {
                    'level': 'genus',
                    'parent': row['Family'] if pd.notna(row['Family']) else None
                }
            
            # 处理科级
            if pd.notna(row['Family']):
                feature_mapping[row['Family']] = {
                    'level': 'family',
                    'parent': row['Order'] if pd.notna(row['Order']) else None
                }
            
            # 处理目级
            if pd.notna(row['Order']):
                feature_mapping[row['Order']] = {
                    'level': 'order',
                    'parent': row['Class'] if pd.notna(row['Class']) else None
                }
            
            # 处理纲级
            if pd.notna(row['Class']):
                feature_mapping[row['Class']] = {
                    'level': 'class',
                    'parent': row['Phylum'] if pd.notna(row['Phylum']) else None
                }
            
            # 处理门级
            if pd.notna(row['Phylum']):
                feature_mapping[row['Phylum']] = {
                    'level': 'phylum',
                    'parent': None
                }
        
        return feature_mapping
    
    @staticmethod
    def process_taxonomy_info(csv_path):
        """Process taxonomy information and return all necessary data structures
        
        Args:
            csv_path: Path to the taxonomy CSV file
            
        Returns:
            Dictionary containing processed taxonomy information
        """
        # Load raw taxonomy data
        taxonomy_df = TaxonomyLoader.load_taxonomy(csv_path)
        
        # Extract taxa lists by level
        taxa_by_level = TaxonomyLoader.extract_taxa_by_level(taxonomy_df)
        
        # Create feature mapping
        feature_mapping = TaxonomyLoader.get_feature_mapping(taxonomy_df)
        
        return {
            'raw_taxonomy': taxonomy_df,
            'taxa_by_level': taxa_by_level,
            'feature_mapping': feature_mapping
        }
    
    @staticmethod
    def print_taxonomy_summary(taxonomy_info):
        """Print summary of taxonomy information
        
        Args:
            taxonomy_info: Processed taxonomy information dictionary
        """
        print("\nTaxonomy Summary:")
        print("-" * 50)
        
        taxa_by_level = taxonomy_info['taxa_by_level']
        for level, taxa in taxa_by_level.items():
            print(f"{level.capitalize()}: {len(taxa)} unique taxa")
        
        print("\nExample taxa at each level:")
        for level, taxa in taxa_by_level.items():
            print(f"\n{level.capitalize()} (first 5):")
            for taxon in sorted(taxa)[:5]:
                print(f"  - {taxon}")
        
        print("\nFeature mapping summary:")
        feature_mapping = taxonomy_info['feature_mapping']
        level_counts = {}
        for feature, info in feature_mapping.items():
            level = info['level']
            level_counts[level] = level_counts.get(level, 0) + 1
            
        for level, count in sorted(level_counts.items()):
            print(f"{level.capitalize()}: {count} mapped features")





def stratified_sample_indices(y, weights, sample_size):
    """分层采样，确保每个类别都被采样到"""
    classes = np.unique(y)
    sampled_indices = []
    
    # 计算每个类别应采样的数量
    class_counts = np.bincount(y)
    class_proportions = class_counts / len(y)
    min_class_proportion = np.min(class_proportions)
    
    for class_label in classes:
        # 获取该类别的样本索引
        class_indices = np.where(y == class_label)[0]
        class_weights = weights[class_indices]
        class_weights = class_weights / np.sum(class_weights)
        
        # 确保每个类别至少采样到一定比例
        class_sample_size = max(
            int(sample_size * class_proportions[class_label]),
            int(sample_size * min_class_proportion * 0.8)  # 保证最小采样量
        )
        
        # 基于权重采样
        sampled_class_indices = np.random.choice(
            class_indices,
            size=class_sample_size,
            p=class_weights,
            replace=True
        )
        sampled_indices.extend(sampled_class_indices)
    
    return np.array(sampled_indices)

def train_evaluate_random_forest(X, y):
    """Initialize and return Random Forest model with specified parameters"""
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

def calculate_specificity(y_true, y_pred, class_labels=None):
    """
    Calculate specificity for each class using one-vs-rest approach
    """
    # 如果没有提供class_labels，则从数据中获取唯一类别数
    if class_labels is None:
        n_classes = len(np.unique(y_true))
    else:
        n_classes = len(class_labels)
        
    specificities = []
    for label in range(n_classes):
        # Convert to binary problem
        y_true_binary = (y_true == label).astype(int)
        y_pred_binary = (y_pred == label).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        # Calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
    
    return np.array(specificities)

def normalize_metric(metric):
    """标准化指标到[0,1]范围"""
    min_val = np.min(metric)
    max_val = np.max(metric)
    if max_val == min_val:
        return np.zeros_like(metric)
    return (metric - min_val) / (max_val - min_val)

def calculate_uncertainty(y_pred_proba, y_true):
    """改进的不确定性计算，考虑类别平衡"""
    n_classes = y_pred_proba.shape[1]
    
    # 基础不确定性指标
    entropy_vals = np.apply_along_axis(entropy, 1, y_pred_proba)
    
    # 计算每个样本预测正确的概率
    true_class_probs = y_pred_proba[np.arange(len(y_true)), y_true]
    
    # 计算预测置信度（与真实类别的关系）
    confidence_diff = 1.0 - true_class_probs
    
    # 计算类间差异（困难样本识别）
    sorted_probs = np.sort(y_pred_proba, axis=1)
    class_gaps = []
    for i in range(n_classes - 1):
        gap = sorted_probs[:, -(i+1)] - sorted_probs[:, -(i+2)]
        class_gaps.append(gap)
    mean_class_gap = np.mean(class_gaps, axis=0)
    
    # 标准化所有指标
    norm_entropy = normalize_metric(entropy_vals)
    norm_confidence = normalize_metric(confidence_diff)
    norm_gap = normalize_metric(mean_class_gap)
    
    # 组合不确定性指标，更注重类别区分
    combined_uncertainty = (
        0.3 * norm_entropy +      # 整体预测不确定性
        0.4 * norm_confidence +   # 与真实类别的差异
        0.3 * norm_gap           # 类别间的区分度
    )
    
    return {
        'entropy': entropy_vals,
        'confidence': confidence_diff,
        'class_gap': mean_class_gap,
        'combined': combined_uncertainty
    }

def calculate_detailed_accuracy_metrics(y_true, y_pred, y_pred_proba):
    """Calculate detailed accuracy metrics including balanced accuracy and per-class accuracy
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        
    Returns:
        Dictionary containing detailed accuracy metrics
    """
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    import numpy as np
    
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Calculate per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Calculate top-k accuracy (k=2,3)
    n_samples = len(y_true)
    top_2_hits = 0
    top_3_hits = 0
    
    for i in range(n_samples):
        true_label = y_true[i]
        top_k_indices = np.argsort(y_pred_proba[i])[::-1]
        
        if true_label in top_k_indices[:2]:
            top_2_hits += 1
        if true_label in top_k_indices[:3]:
            top_3_hits += 1
    
    top_2_accuracy = top_2_hits / n_samples
    top_3_accuracy = top_3_hits / n_samples
    
    accuracy_metrics = {
        'overall_accuracy': overall_accuracy,
        'balanced_accuracy': balanced_acc,
        'per_class_accuracy': per_class_accuracy,
        'top_2_accuracy': top_2_accuracy,
        'top_3_accuracy': top_3_accuracy
    }
    
    return accuracy_metrics

def calculate_metrics(y_true, y_pred, y_pred_proba, class_labels=None):
    """Calculate all metrics including the enhanced accuracy metrics"""
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    
    # Calculate specificity
    specificities = calculate_specificity(y_true, y_pred, class_labels)
    
    # Calculate uncertainty metrics
    uncertainty = calculate_uncertainty(y_pred_proba, y_true)
    
    # Calculate detailed accuracy metrics
    accuracy_metrics = calculate_detailed_accuracy_metrics(y_true, y_pred, y_pred_proba)
    
    overall_metrics = {
        'accuracy': accuracy_metrics['overall_accuracy'],
        'balanced_accuracy': accuracy_metrics['balanced_accuracy'],
        'top_2_accuracy': accuracy_metrics['top_2_accuracy'],
        'top_3_accuracy': accuracy_metrics['top_3_accuracy'],
        'f1_score': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'auc_roc': roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr'),
        'specificity': np.mean(specificities),
        'mean_uncertainty': np.mean(uncertainty['combined'])
    }
    
    # Calculate per-class metrics
    per_class_metrics = {
        'accuracy_per_class': accuracy_metrics['per_class_accuracy'],
        'precision_per_class': precision_score(y_true, y_pred, average=None),
        'recall_per_class': recall_score(y_true, y_pred, average=None),
        'f1_per_class': f1_score(y_true, y_pred, average=None),
        'auc_roc_per_class': roc_auc_score(y_true_bin, y_pred_proba, average=None),
        'specificity_per_class': specificities
    }
    
    if class_labels is not None:
        per_class_metrics = {
            metric_name: {
                class_labels[i]: score for i, score in enumerate(scores)
            }
            for metric_name, scores in per_class_metrics.items()
        }
    
    return {
        'overall': overall_metrics,
        'per_class': per_class_metrics,
        'uncertainty': uncertainty
    }

def calculate_class_weights(y):
    """计算类别权重"""
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return class_weights

def calculate_sampling_weights(uncertainty_metrics, y_train, iteration, max_iterations):
    """改进的采样权重计算，考虑类别平衡"""
    combined_uncertainty = uncertainty_metrics['combined']
    
    # 计算类别权重
    class_weights = calculate_class_weights(y_train)
    sample_class_weights = class_weights[y_train]
    
    # 基础权重（确保类别平衡）
    base_weight = sample_class_weights / np.sum(sample_class_weights)
    
    # 不确定性权重随迭代增加
    uncertainty_ratio = min(0.8, (iteration + 1) / max_iterations)
    certainty_ratio = 1 - uncertainty_ratio
    
    # 结合类别权重和不确定性
    weights = (certainty_ratio * base_weight + 
              uncertainty_ratio * normalize_metric(combined_uncertainty))
    
    # 添加温度参数，随迭代降低
    temperature = max(0.3, 1.0 - 0.7 * (iteration / max_iterations))
    weights = np.exp(weights / temperature)
    
    return weights / np.sum(weights)


def calculate_mean_metrics(fold_metrics):
    """计算所有fold的平均指标"""
    mean_metrics = {
        'overall': {
            metric: np.mean([fold['overall'][metric] for fold in fold_metrics])
            for metric in fold_metrics[0]['overall'].keys()
        }
    }
    
    # 如果存在per_class指标，也计算其平均值
    if 'per_class' in fold_metrics[0]:
        mean_metrics['per_class'] = {}
        for metric in fold_metrics[0]['per_class'].keys():
            mean_metrics['per_class'][metric] = {}
            for class_name in fold_metrics[0]['per_class'][metric].keys():
                values = [fold['per_class'][metric][class_name] for fold in fold_metrics]
                mean_metrics['per_class'][metric][class_name] = np.mean(values)
    
    return mean_metrics

def calculate_std_metrics(fold_metrics):
    """计算所有fold的标准差指标"""
    std_metrics = {
        'overall': {
            metric: np.std([fold['overall'][metric] for fold in fold_metrics])
            for metric in fold_metrics[0]['overall'].keys()
        }
    }
    
    # 如果存在per_class指标，也计算其标准差
    if 'per_class' in fold_metrics[0]:
        std_metrics['per_class'] = {}
        for metric in fold_metrics[0]['per_class'].keys():
            std_metrics['per_class'][metric] = {}
            for class_name in fold_metrics[0]['per_class'][metric].keys():
                values = [fold['per_class'][metric][class_name] for fold in fold_metrics]
                std_metrics['per_class'][metric][class_name] = np.std(values)
    
    return std_metrics

def transform_prediction_probabilities(val_probas_raw, model, X_val, temperature=1.0, smoothing_factor=0.1):
    """
    Transform raw prediction probabilities using feature-based confidence calibration.
    
    Args:
        val_probas_raw (np.ndarray): Raw prediction probabilities from the model
        model: Trained model instance
        X_val: Validation features
        temperature (float): Temperature parameter for softmax scaling
        smoothing_factor (float): Label smoothing factor
    
    Returns:
        np.ndarray: Transformed prediction probabilities
    """
    import numpy as np
    from scipy.special import softmax
    
    # Ensure X_val matches the feature dimensions used in training
    n_features = len(model.feature_importances_)
    if X_val.shape[1] != n_features:
        # Only use the first n_features columns if dimensions don't match
        X_val = X_val[:, :n_features]
    
    # 1. Feature importance based confidence adjustment
    feature_importances = model.feature_importances_
    confidence_scores = np.zeros(len(X_val))
    
    for i in range(len(X_val)):
        # Weight each sample's confidence by feature importance
        confidence_scores[i] = np.sum(X_val[i] * feature_importances)
    
    # Normalize confidence scores
    confidence_scores = (confidence_scores - np.min(confidence_scores)) / \
                       (np.max(confidence_scores) - np.min(confidence_scores))
    
    # 2. Ensemble decision paths analysis
    n_estimators = len(model.estimators_)
    decision_paths = np.zeros((len(X_val), n_estimators))
    
    for i, tree in enumerate(model.estimators_):
        # Get decision path lengths for each sample
        paths = tree.decision_path(X_val)
        decision_paths[:, i] = np.asarray(paths.sum(axis=1)).flatten()
    
    # Normalize decision path lengths
    mean_path_length = np.mean(decision_paths, axis=1)
    path_confidence = 1 / (1 + np.exp(-mean_path_length))  # Sigmoid transformation
    
    # 3. Temperature scaling and probability smoothing
    transformed_probas = np.zeros_like(val_probas_raw)
    
    for i in range(len(X_val)):
        # Apply temperature scaling
        scaled_logits = np.log(val_probas_raw[i] + 1e-10) / temperature
        
        # Apply label smoothing
        smooth_probs = (1 - smoothing_factor) * softmax(scaled_logits) + \
                      smoothing_factor / len(scaled_logits)
        
        # Combine with feature-based confidence
        feature_weight = 0.7  # Weight for feature-based confidence
        path_weight = 0.3    # Weight for decision path confidence
        
        confidence_weight = (feature_weight * confidence_scores[i] + 
                           path_weight * path_confidence[i])
        
        # Final probability transformation
        transformed_probas[i] = (confidence_weight * smooth_probs + 
                               (1 - confidence_weight) * val_probas_raw[i])
        
        # Ensure probabilities sum to 1
        transformed_probas[i] = transformed_probas[i] / np.sum(transformed_probas[i])
    
    return transformed_probas

def adaptive_confidence_calibration(val_probas_raw, y_val, n_bins=10):
    """
    Apply adaptive confidence calibration using histogram binning.
    
    Args:
        val_probas_raw (np.ndarray): Raw prediction probabilities
        y_val: Validation labels
        n_bins (int): Number of bins for calibration
    
    Returns:
        np.ndarray: Calibrated prediction probabilities
    """
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    
    n_classes = val_probas_raw.shape[1]
    calibrated_probas = np.zeros_like(val_probas_raw)
    
    for class_idx in range(n_classes):
        # Get predicted probabilities for current class
        class_probs = val_probas_raw[:, class_idx]
        
        # Create binary labels for current class
        binary_labels = (y_val == class_idx).astype(int)
        
        # Train isotonic regression
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(class_probs, binary_labels)
        
        # Calibrate probabilities
        calibrated_probas[:, class_idx] = ir.predict(class_probs)
    
    # Normalize probabilities to sum to 1
    row_sums = calibrated_probas.sum(axis=1)
    calibrated_probas = calibrated_probas / row_sums[:, np.newaxis]
    
    return calibrated_probas

import tensorflow as tf

class TransformerFeatureProcessor:
    def __init__(self, feature_dim, num_heads=4, num_layers=2):
        self.feature_dim = feature_dim
        self.adjusted_dim = ((feature_dim + num_heads - 1) // num_heads) * num_heads
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model = None
        
    def build_model(self):
        inputs = tf.keras.Input(shape=(self.feature_dim,))
        
        # First dense layer to adjust dimensions
        x = tf.keras.layers.Dense(self.adjusted_dim, activation='relu')(inputs)
        
        # Add positional encoding by treating features as a sequence
        x = tf.keras.layers.Reshape((1, self.adjusted_dim))(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Multiple transformer layers
        for _ in range(self.num_layers):
            # Multi-head attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.adjusted_dim // self.num_heads
            )(x, x)
            
            # Add & Norm
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed forward network
            ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(self.adjusted_dim * 2, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(self.adjusted_dim)
            ])(x)
            
            # Add & Norm
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Final processing
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(self.feature_dim)(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mse')  # Using MSE loss for feature reconstruction
    
    def process_features(self, features, training=False):
        """Process features through the transformer model"""
        if self.model is None:
            self.build_model()
            
        # Convert to tensorflow tensor if necessary
        if not isinstance(features, tf.Tensor):
            features = tf.convert_to_tensor(features, dtype=tf.float32)
            
        # Add batch dimension if necessary
        if len(features.shape) == 1:
            features = tf.expand_dims(features, 0)
            
        return self.model(features, training=training)


def apply_feature_transformation_with_attention(val_probas_raw, model, X_val, X_val_transformed, y_val, 
                                             temperature=1.0, smoothing_factor=0.1):
    """
    Enhanced feature transformation with feature fusion.
    
    Args:
        val_probas_raw: Raw prediction probabilities from Random Forest
        model: Trained Random Forest model
        X_val: Original validation features
        X_val_transformed: Transformer processed features
        y_val: Validation labels
        temperature: Temperature parameter for probability scaling
        smoothing_factor: Label smoothing factor
    
    Returns:
        np.ndarray: Transformed prediction probabilities
    """
    # Ensure all feature arrays have matching first dimensions
    n_samples = len(X_val)
    n_features_rf = len(model.feature_importances_)
    
    # Prepare feature arrays
    X_val_orig = X_val[:, :n_features_rf]  # Use only relevant features for RF
    
    # Combine features for transformation
    combined_features = np.concatenate([
        X_val_orig,  # Original features used by RF
        X_val_transformed,  # Transformer processed features
        val_probas_raw  # Random Forest's prediction probabilities
    ], axis=1)
    
    # Apply original feature transformation with combined features
    transformed_probas = transform_prediction_probabilities(
        val_probas_raw,
        model,
        combined_features,
        temperature=temperature,
        smoothing_factor=smoothing_factor
    )
    
    # Apply confidence calibration
    calibrated_probas = adaptive_confidence_calibration(
        transformed_probas,
        y_val
    )
    
    return calibrated_probas




def cross_validate_model(X, y, class_labels=None, n_iterations=15):
    """改进的交叉验证训练过程，使用特征融合和增强型置信度转换"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    n_classes = len(np.unique(y))
    
    print("\nTraining Random Forest with enhanced confidence calibration...")
    print("-" * 50)
    
    # 初始化Transformer特征处理器
    feature_processor = TransformerFeatureProcessor(
        feature_dim=X.shape[1],
        num_heads=4,
        num_layers=2
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Set different seed for each fold
        set_global_random_seed(42 + fold)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 初始化
        current_weights = np.ones(len(X_train)) / len(X_train)
        best_model = None
        best_score = 0
        best_auc = 0
        
        
        # 渐进式训练
        for iteration in range(n_iterations):
            
            # 训练模型
            model = train_evaluate_random_forest(X_train, y_train)
            model.fit(X_train, y_train)
            
            # 评估当前模型
            y_val_pred = model.predict(X_val)
            y_val_pred_proba = model.predict_proba(X_val)
            
            # 计算多个指标
            val_metrics = calculate_metrics(y_val, y_val_pred, y_val_pred_proba)
            val_score = val_metrics['overall']['accuracy']
            val_auc = val_metrics['overall']['auc_roc']
            
            # 更新最佳模型（考虑AUC-ROC）
            if val_auc > best_auc or (val_auc == best_auc and val_score > best_score):
                best_score = val_score
                best_auc = val_auc
                best_model = model
            
            # 获取预测概率并更新采样权重
            y_pred_proba_train = model.predict_proba(X_train)
            uncertainty_metrics = calculate_uncertainty(y_pred_proba_train, y_train)
            current_weights = calculate_sampling_weights(
                uncertainty_metrics,
                y_train, 
                iteration,
                n_iterations
            )
            
            print(f"Fold {fold}, Iteration {iteration + 1}: "
                  f"Accuracy = {val_score:.3f}, AUC-ROC = {val_auc:.3f}")
        
        if best_model is None:
            raise ValueError("No model was selected as best_model")
        
        # 1. 原始特征
        print("- Using original features")
        X_val_original = X_val.copy()
        
        # 2. Transformer处理后的特征
        print("- Applying Transformer attention")
        X_val_transformed = feature_processor.process_features(X_val).numpy()
        
        # 3. 获取Random Forest的预测概率
        print("- Getting Random Forest predictions")
        val_probas_raw = best_model.predict_proba(X_val)
        
        # 应用增强的特征变换模块（使用三种特征）
        print("\nApplying Enhanced Feature Transform Module...")
        transformed_probas = apply_feature_transformation_with_attention(
            val_probas_raw=val_probas_raw,
            model=best_model,
            X_val=X_val_original,  # 原始特征
            X_val_transformed=X_val_transformed,  # Transformer处理后的特征
            y_val=y_val,
            temperature=1.4,
            smoothing_factor=0.1
        )
        
        # 使用最佳模型进行最终评估
        y_pred = best_model.predict(X_val)
        
        # 6. 计算最终指标
        fold_result = calculate_metrics(y_val, y_pred, transformed_probas, class_labels)
        fold_metrics.append(fold_result)
        
        # 打印详细的结果比较
        print(f"\nFold {fold} Results Comparison:")
        print("Before transformation:")
        metrics_before = calculate_metrics(y_val, best_model.predict(X_val), 
                                        val_probas_raw, class_labels)
        
        print("Overall metrics:")
        print(f"Accuracy: {metrics_before['overall']['accuracy']:.3f}")
        print(f"AUC-ROC: {metrics_before['overall']['auc_roc']:.3f}")
        print(f"F1-Score: {metrics_before['overall']['f1_score']:.3f}")
        
        print("\nAfter transformation with feature fusion:")
        print("Overall metrics:")
        print(f"Accuracy: {fold_result['overall']['accuracy']:.3f}")
        print(f"AUC-ROC: {fold_result['overall']['auc_roc']:.3f}")
        print(f"F1-Score: {fold_result['overall']['f1_score']:.3f}")
        
        # 打印每个类别的性能
        if class_labels:
            print("\nPer-class metrics:")
            print("-" * 50)
            print("Class      AUC-ROC    F1-Score   Before → After")
            for i, class_name in enumerate(class_labels):
                auc_before = metrics_before['per_class']['auc_roc_per_class'][class_name]
                auc_after = fold_result['per_class']['auc_roc_per_class'][class_name]
                f1_before = metrics_before['per_class']['f1_per_class'][class_name]
                f1_after = fold_result['per_class']['f1_per_class'][class_name]
                
                print(f"{class_name:<10} "
                      f"{auc_before:.3f}->{auc_after:.3f}  "
                      f"{f1_before:.3f}->{f1_after:.3f}")
        
        print("-" * 50)
    
    # 计算并返回整体结果
    final_results = {
        'mean_metrics': calculate_mean_metrics(fold_metrics),
        'std_metrics': calculate_std_metrics(fold_metrics),
        'fold_metrics': fold_metrics
    }
    
    # 打印最终汇总结果
    print("\nFinal Results Summary:")
    print("=" * 50)
    mean_metrics = final_results['mean_metrics']
    std_metrics = final_results['std_metrics']
    
    print("Overall Performance:")
    for metric in ['accuracy', 'auc_roc', 'f1_score']:
        mean_val = mean_metrics['overall'][metric]
        std_val = std_metrics['overall'][metric]
        print(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")
    
    if class_labels:
        print("\nPer-class Performance:")
        for class_name in class_labels:
            print(f"\n{class_name}:")
            for metric in ['auc_roc_per_class', 'f1_per_class']:
                mean_val = mean_metrics['per_class'][metric][class_name]
                std_val = std_metrics['per_class'][metric][class_name]
                metric_name = metric.replace('_per_class', '')
                print(f"{metric_name}: {mean_val:.3f} ± {std_val:.3f}")
    
    return final_results


def print_final_results(results, class_labels=None):
    """Print final results with enhanced accuracy metrics"""
    mean_metrics = results['mean_metrics']
    std_metrics = results['std_metrics']
    
    print("\nFinal Random Forest Results with Uncertainty-based Sampling:")
    print("-" * 50)
    print(f"Overall Accuracy: {mean_metrics['overall']['accuracy']*100:.1f}% ± {std_metrics['overall']['accuracy']*100:.1f}%")
    print(f"Balanced Accuracy: {mean_metrics['overall']['balanced_accuracy']*100:.1f}% ± {std_metrics['overall']['balanced_accuracy']*100:.1f}%")
    print(f"Top-2 Accuracy: {mean_metrics['overall']['top_2_accuracy']*100:.1f}% ± {std_metrics['overall']['top_2_accuracy']*100:.1f}%")
    print(f"Top-3 Accuracy: {mean_metrics['overall']['top_3_accuracy']*100:.1f}% ± {std_metrics['overall']['top_3_accuracy']*100:.1f}%")
    print(f"F1-Score: {mean_metrics['overall']['f1_score']:.3f} ± {std_metrics['overall']['f1_score']:.3f}")
    print(f"AUC-ROC: {mean_metrics['overall']['auc_roc']:.3f} ± {std_metrics['overall']['auc_roc']:.3f}")
    print(f"Precision: {mean_metrics['overall']['precision']:.3f} ± {std_metrics['overall']['precision']:.3f}")
    print(f"Recall: {mean_metrics['overall']['recall']:.3f} ± {std_metrics['overall']['recall']:.3f}")
    print(f"Specificity: {mean_metrics['overall']['specificity']:.3f} ± {std_metrics['overall']['specificity']:.3f}")
    print(f"Mean Uncertainty: {mean_metrics['overall']['mean_uncertainty']:.3f} ± {std_metrics['overall']['mean_uncertainty']:.3f}")
    
    if class_labels:
        print("\nDisease-Specific Performance Analysis:")
        print("-" * 100)
        print("Disease        Accuracy   Precision  Recall    F1-Score  AUC-ROC   Specificity")
        print("-" * 100)
        
        for class_name in class_labels:
            accuracy = mean_metrics['per_class']['accuracy_per_class'][class_name]
            accuracy_std = std_metrics['per_class']['accuracy_per_class'][class_name]
            
            precision = mean_metrics['per_class']['precision_per_class'][class_name]
            precision_std = std_metrics['per_class']['precision_per_class'][class_name]
            
            recall = mean_metrics['per_class']['recall_per_class'][class_name]
            recall_std = std_metrics['per_class']['recall_per_class'][class_name]
            
            f1 = mean_metrics['per_class']['f1_per_class'][class_name]
            f1_std = std_metrics['per_class']['f1_per_class'][class_name]
            
            auc = mean_metrics['per_class']['auc_roc_per_class'][class_name]
            auc_std = std_metrics['per_class']['auc_roc_per_class'][class_name]
            
            specificity = mean_metrics['per_class']['specificity_per_class'][class_name]
            specificity_std = std_metrics['per_class']['specificity_per_class'][class_name]
            
            print(f"{class_name:<12} {accuracy:.3f}±{accuracy_std:.3f}  "
                  f"{precision:.3f}±{precision_std:.3f}  "
                  f"{recall:.3f}±{recall_std:.3f}  "
                  f"{f1:.3f}±{f1_std:.3f}  "
                  f"{auc:.3f}±{auc_std:.3f}  "
                  f"{specificity:.3f}±{specificity_std:.3f}")
            
def convert_numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    return obj

def save_results(results):
    """Save results to JSON file with numpy array conversion"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('outputs/results')
    output_file = output_dir / f'MASFI/HFEM+MFFM+FIM+USM+FTM_results_{timestamp}.json'
    
    # Convert numpy arrays to Python lists before saving
    converted_results = convert_numpy_to_python(results)
    
    with open(output_file, 'w') as f:
        json.dump(converted_results, f, indent=4)
    
    print(f"\nResults saved to: {output_file}")
    
class TaxonomyBasedFeatureProcessor:
    """Process features using taxonomy information to enhance representation"""
    
    def __init__(self, taxonomy_info, feature_names):
        """
        Initialize the feature processor with taxonomy information
        
        Args:
            taxonomy_info (dict): Processed taxonomy information
            feature_names (list): List of feature names corresponding to input data
        """
        self.taxonomy_info = taxonomy_info
        self.feature_names = feature_names
        self.feature_mapping = taxonomy_info['feature_mapping']
        self.taxa_by_level = taxonomy_info['taxa_by_level']
        
        # Create reverse mappings for quick lookups
        self._create_reverse_mappings()
        
    def _create_reverse_mappings(self):
        """Create reverse mappings for efficient lookups"""
        # Map from taxa to their children at each level
        self.level_children = {
            'phylum': {},
            'class': {},
            'order': {},
            'family': {},
            'genus': {}
        }
        
        # Create hierarchical mappings
        for feature, info in self.feature_mapping.items():
            level = info['level']
            parent = info['parent']
            
            if parent:
                if parent not in self.level_children[level]:
                    self.level_children[level][parent] = set()
                self.level_children[level][parent].add(feature)
    
    def get_hierarchical_features(self, X):
        """
        Generate hierarchical features based on taxonomy using iterative approach
        
        Args:
            X (np.ndarray): Input feature matrix
            
        Returns:
            np.ndarray: Enhanced feature matrix with hierarchical information
        """
        import numpy as np
        
        n_samples = X.shape[0]
        enhanced_features = []
        
        # Process each taxonomic level
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            level_features = np.zeros((n_samples, len(self.taxa_by_level[level])))
            
            for i, taxon in enumerate(self.taxa_by_level[level]):
                # Get feature indices for current taxon and its descendants
                feature_indices = self._get_taxon_feature_indices(taxon, level)
                
                if feature_indices:
                    # Aggregate features
                    level_features[:, i] = X[:, feature_indices].sum(axis=1)
            
            enhanced_features.append(level_features)
        
        # Concatenate all hierarchical features
        return np.concatenate(enhanced_features, axis=1)
    
    def _get_taxon_feature_indices(self, taxon, level):
        """
        Get feature indices for a taxon and its descendants using iterative approach
        """
        feature_indices = []
        
        # Direct features of this taxon
        for i, feature in enumerate(self.feature_names):
            if feature in self.feature_mapping:
                if (self.feature_mapping[feature]['level'] == level and 
                    (feature == taxon or self.feature_mapping[feature]['parent'] == taxon)):
                    feature_indices.append(i)
        
        return feature_indices
    
    def get_taxonomic_weights(self):
        """
        Generate feature weights based on taxonomic level
        
        Returns:
            np.ndarray: Weight vector for features
        """
        import numpy as np
        
        # Define level weights (higher weight for more specific levels)
        level_weights = {
            'phylum': 0.6,
            'class': 0.7,
            'order': 0.8,
            'family': 0.9,
            'genus': 1.0
        }
        
        weights = np.ones(len(self.feature_names))
        
        for i, feature in enumerate(self.feature_names):
            if feature in self.feature_mapping:
                level = self.feature_mapping[feature]['level']
                weights[i] = level_weights[level]
        
        return weights
    
    def get_feature_relationships(self):
        """
        Get feature relationship matrix based on taxonomy using iterative approach
        
        Returns:
            np.ndarray: Feature relationship matrix
        """
        import numpy as np
        
        n_features = len(self.feature_names)
        relationships = np.zeros((n_features, n_features))
        
        # Pre-compute taxonomic paths for efficiency
        feature_paths = {}
        for feature in self.feature_names:
            if feature in self.feature_mapping:
                feature_paths[feature] = self._get_taxonomic_path_iterative(feature)
        
        for i, feature1 in enumerate(self.feature_names):
            for j, feature2 in enumerate(self.feature_names):
                if i == j:
                    relationships[i, j] = 1.0
                else:
                    # Calculate relationship based on taxonomic distance
                    relationship = self._calculate_taxonomic_relationship_iterative(
                        feature1, feature2, feature_paths
                    )
                    relationships[i, j] = relationship
        
        return relationships
    
    def _get_taxonomic_path_iterative(self, feature):
        """Get taxonomic path using iterative approach"""
        if feature not in self.feature_mapping:
            return []
            
        path = []
        current = feature
        visited = set()  # Prevent cycles
        
        while current in self.feature_mapping and current not in visited:
            visited.add(current)
            path.append(current)
            current = self.feature_mapping[current]['parent']
            if current is None:
                break
                
        return list(reversed(path))
    
    def _calculate_taxonomic_relationship_iterative(self, feature1, feature2, feature_paths):
        """Calculate relationship strength between features using pre-computed paths"""
        if feature1 not in self.feature_mapping or feature2 not in self.feature_mapping:
            return 0.0
            
        path1 = feature_paths.get(feature1, [])
        path2 = feature_paths.get(feature2, [])
        
        # Find common ancestor level
        common_steps = 0
        for a, b in zip(path1, path2):
            if a == b:
                common_steps += 1
            else:
                break
        
        max_path_length = max(len(path1), len(path2))
        if max_path_length == 0:
            return 0.0
            
        return common_steps / max_path_length
    
    def process_features(self, X):
        """
        Process input features using all available taxonomic information
        
        Args:
            X (np.ndarray): Input feature matrix
            
        Returns:
            np.ndarray: Processed feature matrix with taxonomic information
        """
        import numpy as np
        
        print("Generating hierarchical features...")
        hierarchical_features = self.get_hierarchical_features(X)
        
        print("Applying taxonomic weights...")
        weights = self.get_taxonomic_weights()
        weighted_features = X * weights.reshape(1, -1)
        
        print("Computing feature relationships...")
        relationships = self.get_feature_relationships()
        relationship_features = np.dot(X, relationships)
        
        print("Combining feature representations...")
        processed_features = np.concatenate([
            # weighted_features,
            hierarchical_features,
            # relationship_features
        ], axis=1)
        
        return processed_features
def main():
    """Main program entry with enhanced taxonomy-based feature processing"""
    try:
        # Set global random seed
        set_global_random_seed(42)

        # Create directories
        setup_directories()

        # Read data
        print("Reading data...")
        df = pd.read_csv('data/Merged_Combined.csv')

        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        print("Preprocessing data...")
        
        # Extract features and labels
        X = df.iloc[:, :-3]
        feature_names = X.columns.tolist()
        y = df.iloc[:, -1]  # AD/PD/ASD/Control
        
        # Load taxonomy information
        print("\nLoading taxonomy information...")
        taxonomy_path = 'data/taxonomy.csv'
        taxonomy_loader = TaxonomyLoader()
        taxonomy_info = taxonomy_loader.process_taxonomy_info(taxonomy_path)
        
        # Print taxonomy summary
        taxonomy_loader.print_taxonomy_summary(taxonomy_info)
        
        # Initialize taxonomy-based feature processor
        print("\nInitializing taxonomy-based feature processor...")
        feature_processor = TaxonomyBasedFeatureProcessor(taxonomy_info, feature_names)
        
        # Normalize original features
        print("Normalizing features...")
        normalizer = Normalizer(norm='l1')
        X_normalized = normalizer.fit_transform(X)
        
        # Process features using taxonomy information
        print("Processing features using taxonomy information...")
        X_processed = feature_processor.process_features(X_normalized)
        
        # Normalize processed features
        X_processed_normalized = normalizer.fit_transform(X_processed)
        
        # Print feature processing summary
        print(f"\nFeature processing summary:")
        print(f"Original features shape: {X_normalized.shape}")
        print(f"Processed features shape: {X_processed_normalized.shape}")
        
        # Calculate feature importance weights based on taxonomy
        tax_weights = feature_processor.get_taxonomic_weights()
        print("\nTaxonomic level distribution in features:")
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            level_count = sum(1 for f in feature_names 
                            if f in taxonomy_info['feature_mapping'] 
                            and taxonomy_info['feature_mapping'][f]['level'] == level)
            print(f"{level.capitalize()}: {level_count} features")

        # Encode labels
        labelencoder = LabelEncoder()
        y_encoded = labelencoder.fit_transform(y)
        
        # Get class labels
        class_labels = list(labelencoder.classes_)
        
        # Print disease distribution
        print("\nDisease Distribution:")
        print("-" * 30)
        disease_counts = pd.Series(y).value_counts()
        disease_percentages = pd.Series(y).value_counts(normalize=True) * 100
        for class_name, count in disease_counts.items():
            percentage = disease_percentages[class_name]
            print(f"{class_name}: {count} samples ({percentage:.2f}%)")
            
        # Train and evaluate model with uncertainty-based sampling
        print("\nStarting Random Forest training with taxonomy-enhanced features...")
        X_processed_normalized = np.concatenate([X_normalized, X_processed_normalized], axis=1)
        results = cross_validate_model(X_processed_normalized, y_encoded, class_labels)

        # Print and save results
        print_final_results(results, class_labels)
        save_results(results)

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()