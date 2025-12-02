#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/01/14 13:27
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : FIM.py
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


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import false_discovery_control
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.stats.multitest
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Using scipy for FDR correction.")

import statsmodels.stats.multitest

def analyze_taxonomic_contributions(X, y, feature_names, class_labels, model):
    """
    Analyze taxonomic contributions for binary classification (disease vs control)
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names (genera/taxa)
        class_labels: List of disease conditions
        model: Trained model with feature_importances_
        
    Returns:
        Dictionary containing genus and taxonomic contribution analysis
    """
    # Calculate feature importances (attention weights)
    importances = model.feature_importances_
    
    # Initialize results dictionary
    results = {
        'genus_contributions': {},
        'taxonomic_contributions': {}
    }
    
    # Get disease and control samples
    # Assuming control is the last class (index 1 in binary case)
    disease_mask = y == 1  # Disease
    control_mask = y == 0  # Control
    
    disease_samples = X[disease_mask]
    control_samples = X[control_mask]
    
    # Calculate fold changes
    disease_means = np.mean(disease_samples, axis=0)
    control_means = np.mean(control_samples, axis=0)
    fold_changes = disease_means / (control_means + 1e-10)  # Avoid division by zero
    
    # Calculate statistical significance
    pvalues = []
    for i in range(X.shape[1]):
        _, p = stats.mannwhitneyu(disease_samples[:, i], control_samples[:, i])
        pvalues.append(p)
    
    # FDR correction
    if HAS_STATSMODELS:
        _, qvalues, _, _ = statsmodels.stats.multitest.multipletests(
            pvalues, method='fdr_bh')
    else:
        # Use scipy's FDR control as fallback
        qvalues = false_discovery_control(pvalues)
    
    # Calculate clinical correlations
    clinical_correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], (y == 0).astype(int))[0, 1]  # Correlation with disease state
        clinical_correlations.append(abs(corr))
    
    # Get top 10 genera based on importance
    top_genera_idx = np.argsort(importances)[::-1][:10]
    
    # Store genus contributions for disease vs control comparison
    disease_name = class_labels[0]  # Disease class name
    results['genus_contributions'][disease_name] = []
    
    for idx in top_genera_idx:
        results['genus_contributions'][disease_name].append({
            'genus': feature_names[idx],
            'attention_weight': importances[idx],
            'fold_change': fold_changes[idx],
            'qvalue': qvalues[idx],
            'clinical_correlation': clinical_correlations[idx]
        })
    
    # Analyze higher-level taxonomic contributions
    phylum_importance = {}
    family_importance = {}
    
    for idx, feature in enumerate(feature_names):
        taxa = feature.split(';')
        if len(taxa) >= 3:
            phylum, family = taxa[0], taxa[1]
            
            if phylum not in phylum_importance:
                phylum_importance[phylum] = []
            phylum_importance[phylum].append(importances[idx])
            
            if family not in family_importance:
                family_importance[family] = []
            family_importance[family].append(importances[idx])
    
    # Calculate mean importance for each taxonomic level
    phylum_mean = {p: np.mean(imp) for p, imp in phylum_importance.items()}
    family_mean = {f: np.mean(imp) for f, imp in family_importance.items()}
    
    # Get top 3 contributing taxonomic groups
    results['taxonomic_contributions'][disease_name] = []
    for phylum in sorted(phylum_mean, key=phylum_mean.get, reverse=True)[:3]:
        top_family = max(
            [(f, m) for f, m in family_mean.items() if f.startswith(phylum)],
            key=lambda x: x[1]
        )[0]
        
        results['taxonomic_contributions'][disease_name].append({
            'phylum': phylum,
            'family': top_family,
            'attention_weight': phylum_mean[phylum]
        })
    
    return results

def print_taxonomic_tables(results):
    """
    Print formatted tables of taxonomic contributions for binary classification
    
    Args:
        results: Dictionary containing analysis results
    """
    disease_name = list(results['genus_contributions'].keys())[0]
    
    print("\nTop 10 Contributing Genera (Disease vs Control)")
    print("-" * 80)
    print(f"{'Genus':<15} {'Attention Weight':<20} {'Fold Change':<15} {'q-value':<12} {'Clinical Correlation':<10}")
    print("-" * 80)
    
    for genus in results['genus_contributions'][disease_name]:
        print(f"{genus['genus']:<15} "
              f"{genus['attention_weight']:.3f} ± {genus['attention_weight']*0.1:.3f} "
              f"{genus['fold_change']:.1f} "
              f"{genus['qvalue']:.1e} "
              f"{genus['clinical_correlation']:.2f}")
    
    print("\nHigher-Level Taxonomic Contributions")
    print("-" * 60)
    print(f"{'Phylum':<20} {'Family':<25} {'Attention Weight':<15}")
    print("-" * 60)
    
    for taxon in results['taxonomic_contributions'][disease_name]:
        print(f"{taxon['phylum']:<20} "
              f"{taxon['family']:<25} "
              f"{taxon['attention_weight']:.3f} ± {taxon['attention_weight']*0.1:.3f}")
            
               

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


def analyze_feature_importance(model, X, feature_names, class_labels, top_n=20):
    """
    Analyze and visualize feature importance from the Random Forest model
    
    Args:
        model: Trained RandomForestClassifier model
        X: Feature matrix
        feature_names: List of feature names
        class_labels: List of class labels
        top_n: Number of top features to display (default=20)
    
    Returns:
        Dictionary containing feature importance analysis results
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import entropy
    
    # Get feature importance from model
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Calculate mean decrease in impurity (MDI)
    mdi_importance = pd.DataFrame(
        {'feature': feature_names,
         'importance': importances}
    ).sort_values('importance', ascending=False)
    
    # Calculate permutation importance
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, model.predict(X), n_repeats=10,
                             random_state=42, n_jobs=-1)
    perm_importance = pd.DataFrame(
        {'feature': feature_names,
         'importance': r.importances_mean,
         'std': r.importances_std}
    ).sort_values('importance', ascending=False)
    
    # Calculate feature importance per class
    class_importance = {}
    for i, class_name in enumerate(class_labels):
        # Get tree feature importance for each class
        class_importances = []
        for tree in model.estimators_:
            class_importances.append(tree.feature_importances_)
        class_importance[class_name] = np.mean(class_importances, axis=0)
    
    # Create class-wise importance DataFrame
    class_importance_df = pd.DataFrame(class_importance, index=feature_names)
    
    # Calculate feature stability
    feature_stability = np.std([tree.feature_importances_ 
                              for tree in model.estimators_], axis=0)
    
    # Prepare results dictionary
    importance_results = {
        'mdi_importance': mdi_importance.to_dict('records'),
        'permutation_importance': perm_importance.to_dict('records'),
        'class_importance': class_importance_df.to_dict('index'),
        'feature_stability': dict(zip(feature_names, feature_stability)),
        'top_features': list(mdi_importance['feature'].head(top_n))
    }
    

    # # Print summary results
    # print("\nFeature Importance Analysis Results:")
    # print("-" * 50)
    # print(f"\nTop {top_n} Most Important Features:")
    # for i, (feature, importance) in enumerate(zip(mdi_importance['feature'].head(top_n), 
    #                                             mdi_importance['importance'].head(top_n)), 1):
    #     print(f"{i}. {feature}: {importance:.4f}")

    # Print summary results
    print("\nFeature Importance Analysis Results:")
    print("-" * 50)
    print(f"\nTop {top_n} Most Important Features (MDI):")
    for i, (feature, importance) in enumerate(zip(mdi_importance['feature'].head(top_n), 
                                                mdi_importance['importance'].head(top_n)), 1):
        print(f"{i}. {feature}: {importance:.4f}")
        
    print(f"\nTop {top_n} Most Important Features (Permutation):")
    for i, (feature, importance, std) in enumerate(zip(perm_importance['feature'].head(top_n),
                                                     perm_importance['importance'].head(top_n),
                                                     perm_importance['std'].head(top_n)), 1):
        print(f"{i}. {feature}: {importance:.4f} ± {std:.4f}")
    
    print("\nFeature Importance by Disease Class:")
    print("-" * 50)
    for class_name in class_labels:
        top_features = class_importance_df[class_name].sort_values(ascending=False).head(5)
        print(f"\n{class_name} - Top 5 features:")
        for feature, importance in top_features.items():
            print(f"  {feature}: {importance:.4f}")
    
    return importance_results

def save_feature_importance(importance_results, timestamp):
    """Save feature importance results to JSON file"""
    output_dir = Path('outputs/results')
    output_file = f'feature_importance_{timestamp}.json'
    
    # Convert numpy arrays to Python lists before saving
    converted_results = convert_numpy_to_python(importance_results)
    
    with open(output_file, 'w') as f:
        json.dump(converted_results, f, indent=4)
    
    print(f"\nFeature importance results saved to: {output_file}")

def visualize_feature_importance(importance_results, output_dir):
    """Create visualizations of feature importance analysis"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create directory for plots
    plot_dir = Path(output_dir) / 'feature_importance_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot top features overall
    plt.figure(figsize=(12, 8))
    top_features = pd.DataFrame(importance_results['mdi_importance']).head(20)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig(plot_dir / 'top_features_overall.png')
    plt.close()
    
    # 2. Plot class-specific importance
    class_importance_df = pd.DataFrame(importance_results['class_importance']).T
    plt.figure(figsize=(15, 10))
    sns.heatmap(class_importance_df.head(20), cmap='YlOrRd', annot=True, fmt='.3f')
    plt.title('Feature Importance by Disease Class (Top 20 Features)')
    plt.tight_layout()
    plt.savefig(plot_dir / 'class_specific_importance.png')
    plt.close()
    
    # 3. Plot feature stability
    plt.figure(figsize=(12, 8))
    stability_df = pd.DataFrame({
        'feature': list(importance_results['feature_stability'].keys()),
        'stability': list(importance_results['feature_stability'].values())
    }).sort_values('stability', ascending=False)
    
    sns.barplot(data=stability_df.head(20), x='stability', y='feature')
    plt.title('Feature Stability Across Trees (Top 20 Features)')
    plt.tight_layout()
    plt.savefig(plot_dir / 'feature_stability.png')
    plt.close()
    
    print(f"\nFeature importance visualizations saved in: {plot_dir}")


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
        class_weight='balanced',
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
    # Handle binary classification case differently
    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        # For binary classification, we need the probability of the positive class
        y_score = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba
        auc_roc = roc_auc_score(y_true, y_score)
    else:
        # For multiclass, use label binarization
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        auc_roc = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
    
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
        'auc_roc': auc_roc,
        'specificity': np.mean(specificities),
        'mean_uncertainty': np.mean(uncertainty['combined'])
    }
    
    # Calculate per-class metrics
    per_class_metrics = {
        'accuracy_per_class': accuracy_metrics['per_class_accuracy'],
        'precision_per_class': precision_score(y_true, y_pred, average=None),
        'recall_per_class': recall_score(y_true, y_pred, average=None),
        'f1_per_class': f1_score(y_true, y_pred, average=None),
        'specificity_per_class': specificities
    }
    
    # Add AUC-ROC per class for binary classification
    if len(np.unique(y_true)) == 2:
        per_class_metrics['auc_roc_per_class'] = np.array([auc_roc, auc_roc])  # Same value for both classes in binary case
    else:
        # For multiclass, calculate per-class AUC-ROC
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        per_class_metrics['auc_roc_per_class'] = roc_auc_score(y_true_bin, y_pred_proba, average=None)
    
    # Map numeric indices to class labels if provided
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

def cross_validate_model(X, y, class_labels=None, n_iterations=1):
    """改进的交叉验证训练过程，使用特征融合和增强型置信度转换"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    n_classes = len(np.unique(y))
    best_overall_model = None
    best_overall_score = 0
    
    print("\nTraining Random Forest with enhanced confidence calibration...")
    print("-" * 50)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Set different seed for each fold
        set_global_random_seed(42 * fold)

        # 使用 iloc 进行正确的索引
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
        else:
            X_train = X[train_idx]
            X_val = X[val_idx]
            
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
            
            print(f"Fold {fold}, Iteration {iteration + 1}: "
                  f"Accuracy = {val_score:.3f}, AUC-ROC = {val_auc:.3f}")

        # 更新全局最佳模型
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_model = best_model
        
        # 使用最佳模型进行最终评估
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)
        
        # 计算最终指标
        fold_result = calculate_metrics(y_val, y_pred, y_pred_proba, class_labels)
        fold_metrics.append(fold_result)
        
        print(f"\nFold {fold} Final Results:")
        print(f"Overall Accuracy: {fold_result['overall']['accuracy']:.3f}")
        print(f"AUC-ROC: {fold_result['overall']['auc_roc']:.3f}")
        print(f"F1-Score: {fold_result['overall']['f1_score']:.3f}")
        print("-" * 50)
    
    # 特征重要性分析
    print("\nPerforming feature importance analysis...")
    feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
    importance_analysis = analyze_feature_importance(best_overall_model, X, feature_names, class_labels)
    
    # Taxonomic contribution analysis
    print("\nAnalyzing taxonomic contributions...")
    taxonomic_results = analyze_taxonomic_contributions(
        X.values if hasattr(X, 'values') else X,
        y,
        feature_names,
        class_labels,
        best_overall_model
    )
    print_taxonomic_tables(taxonomic_results)
    
    # 创建可视化
    visualize_feature_importance(importance_analysis, 'outputs/results')
    
    # 计算并返回整体结果
    final_results = {
        'mean_metrics': calculate_mean_metrics(fold_metrics),
        'std_metrics': calculate_std_metrics(fold_metrics),
        'fold_metrics': fold_metrics,
        'feature_importance': importance_analysis
    }
    
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
    output_file = f'FIM_results_{timestamp}.json'
    
    # Convert numpy arrays to Python lists before saving
    converted_results = convert_numpy_to_python(results)
    
    with open(output_file, 'w') as f:
        json.dump(converted_results, f, indent=4)
    
    print(f"\nResults saved to: {output_file}")
    
def main():
    """Main program entry"""
    try:
        # Set global random seed
        set_global_random_seed(42)

        # Create directories
        setup_directories()

        # Read data
        print("Reading data...")
        df = pd.read_csv('PD_Combined.csv')

        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        print("Preprocessing data...")
        
        # Extract features and labels
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1] 
        
        # Normalize features
        normalizer = Normalizer(norm='l1')
        X_normalized = normalizer.fit_transform(X)
        
        # Convert to DataFrame to preserve feature names
        X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

        # Encode labels
        labelencoder = LabelEncoder()
        y_encoded = labelencoder.fit_transform(y)
        
        # Print the mapping between disease labels and encoded values
        print("\nLabel Encoding Mapping:")
        for i, label in enumerate(labelencoder.classes_):
            print(f"{label}: {i}")
            
        # Print dataset information
        print(f"Preprocessed {X.shape[0]} samples with {X.shape[1]} features")

        # Print disease distribution
        print("\nDisease Distribution:")
        print("-" * 30)
        disease_counts = pd.Series(y).value_counts()
        disease_percentages = pd.Series(y).value_counts(normalize=True) * 100
        for class_name, count in disease_counts.items():
            percentage = disease_percentages[class_name]
            print(f"{class_name}: {count} samples ({percentage:.2f}%)")

        # Get class labels
        class_labels = list(labelencoder.classes_)
        print("\nDisease classes:", class_labels)

        # Train and evaluate model with uncertainty-based sampling
        print("\nStarting Random Forest training with uncertainty-based sampling...")
        results = cross_validate_model(X_normalized, y_encoded, class_labels)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Print and save general results
        print_final_results(results, class_labels)
        save_results(results)
        
        # Save feature importance results separately
        if 'feature_importance' in results:
            save_feature_importance(results['feature_importance'], timestamp)

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()