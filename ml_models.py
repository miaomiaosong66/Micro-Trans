#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/01/09 10:44
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : ml_models.py
# @Description   : 


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import xgboost as xgb
import numpy as np
from pathlib import Path
import json
from datetime import datetime



# Add CNN-specific imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


import random
import os
import numpy as np


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

def train_evaluate_transformer(X, y):
    """
    Transformer implementation for tabular data classification:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Layer normalization
    - Dropout for regularization
    """
    import tensorflow as tf
    from tensorflow.keras import layers
    import numpy as np

    # Disable mixed precision to ensure consistent float32 usage
    tf.keras.mixed_precision.set_global_policy('float32')
    
    # Set TensorFlow to use deterministic algorithms
    tf.config.experimental.enable_op_determinism()
    
    class MultiHeadSelfAttention(layers.Layer):
        def __init__(self, embed_dim, num_heads=8):
            super(MultiHeadSelfAttention, self).__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            
            assert (
                self.head_dim * num_heads == embed_dim
            ), "Embedding dimension needs to be divisible by number of heads"
            
            self.query_dense = layers.Dense(embed_dim, dtype='float32')
            self.key_dense = layers.Dense(embed_dim, dtype='float32')
            self.value_dense = layers.Dense(embed_dim, dtype='float32')
            self.combine_heads = layers.Dense(embed_dim, dtype='float32')
            
        def attention(self, query, key, value):
            score = tf.matmul(query, key, transpose_b=True)
            dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
            scaled_score = score / tf.math.sqrt(dim_key)
            weights = tf.nn.softmax(scaled_score, axis=-1)
            output = tf.matmul(weights, value)
            return output
            
        def separate_heads(self, x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
            return tf.transpose(x, perm=[0, 2, 1, 3])
            
        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            
            query = self.query_dense(inputs)
            key = self.key_dense(inputs)
            value = self.value_dense(inputs)
            
            query = self.separate_heads(query, batch_size)
            key = self.separate_heads(key, batch_size)
            value = self.separate_heads(value, batch_size)
            
            attention = self.attention(query, key, value)
            attention = tf.transpose(attention, perm=[0, 2, 1, 3])
            concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
            
            output = self.combine_heads(concat_attention)
            return output



    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = MultiHeadSelfAttention(embed_dim, num_heads)
            self.ffn = tf.keras.Sequential([
                layers.Dense(ff_dim, activation="relu", dtype='float32'),
                layers.Dense(embed_dim, dtype='float32'),
            ])
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, dtype='float32')
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, dtype='float32')
            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)
            
        def call(self, inputs, training=False):
            # Ensure inputs are float32
            inputs = tf.cast(inputs, tf.float32)
            attn_output = self.att(inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)
        

    class TabularTransformer:
        def __init__(self, input_shape, num_classes):
            self.model = self._build_model(input_shape, num_classes)
            self.num_classes = num_classes
        
        def _build_model(self, input_shape, num_classes):
            embed_dim = 32  # Embedding dimension
            num_heads = 4   # Number of attention heads
            ff_dim = 64     # Feed-forward network dimension
            
            inputs = layers.Input(shape=input_shape)
            
            # Reshape inputs to add sequence dimension
            x = layers.Reshape((input_shape[0], 1))(inputs)
            
            # Project to embedding dimension
            x = layers.Dense(embed_dim)(x)
            
            # Add positional encoding with explicit float32 casting
            positions = tf.range(start=0, limit=input_shape[0], delta=1, dtype=tf.float32)
            pos_encoding = tf.cast(self._positional_encoding(input_shape[0], embed_dim), dtype=tf.float32)
            x = x + pos_encoding

            # Transformer blocks
            x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
            x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
            
            # Global average pooling
            x = layers.GlobalAveragePooling1D()(x)
            
            # Final classification layers
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.1)(x)
            outputs = layers.Dense(num_classes, activation="softmax")(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            return model
        

        def _positional_encoding(self, position, d_model):
            angle_rads = self._get_angles(
                np.arange(position)[:, np.newaxis],
                np.arange(d_model)[np.newaxis, :],
                d_model
            )
            
            # Apply sin to even indices in the array
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            
            # Apply cos to odd indices in the array
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            
            pos_encoding = angle_rads[np.newaxis, ...]
            
            # Ensure float32 type
            return tf.cast(pos_encoding, dtype=tf.float32)
        
        def _get_angles(self, pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
            return pos * angle_rates
        
        def fit(self, X, y, **kwargs):
            # Convert labels to categorical
            y_cat = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
            
            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train the model
            self.model.fit(
                X, 
                y_cat,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            return self
        
        def predict(self, X):
            y_pred_proba = self.model.predict(X, verbose=0)
            return np.argmax(y_pred_proba, axis=1)
        
        def predict_proba(self, X):
            return self.model.predict(X, verbose=0)
    
    # Get number of features and classes
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    
    # Create and return the Transformer classifier
    return TabularTransformer(input_shape=(num_features,), num_classes=num_classes)

def configure_gpu():
    """
    Configure GPU settings for optimal training performance
    """
    import tensorflow as tf
    
    # Allow memory growth to prevent TF from taking all GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set TensorFlow to use the first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            # Enable mixed precision training for better performance
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print("GPU is configured successfully")
            print(f"Number of GPUs available: {len(gpus)}")
            print(f"Using GPU: {tf.config.get_visible_devices('GPU')[0].name}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU devices found. Training will proceed on CPU.")

def train_evaluate_cnn(X, y):
    """
    GPU-optimized CNN implementation for tabular data classification
    """
    import tensorflow as tf
    
    # Configure GPU settings
    configure_gpu()
    
    # Set TensorFlow to use deterministic algorithms
    tf.config.experimental.enable_op_determinism()

    def create_cnn_model(input_shape, num_classes):
        # Use float16 for faster GPU computation while keeping float32 for specific layers
        model = tf.keras.Sequential([
            # Input layer - keep as float32
            tf.keras.layers.Input(shape=input_shape, dtype='float32'),
            
            # Conv1D blocks with float16
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Flatten(),
            
            # Dense layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            
            # Output layer - keep as float32 for stability
            tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
        ])
        
        # Use Adam optimizer with mixed precision
        optimizer = tf.keras.optimizers.Adam()
        if tf.config.list_physical_devices('GPU'):
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    class CNNClassifier:
        def __init__(self, input_shape, num_classes):
            self.model = create_cnn_model(input_shape, num_classes)
            self.num_classes = num_classes
        
        def fit(self, X, y, **kwargs):
            # Reshape X for CNN input (samples, timesteps, features)
            X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Convert labels to categorical
            y_cat = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
            
            # Early stopping and model checkpointing
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Train the model with GPU optimizations
            self.model.fit(
                X_reshaped, 
                y_cat,
                epochs=100,
                batch_size=32,  # Adjust based on GPU memory
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            return self
        
        def predict(self, X):
            X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
            y_pred_proba = self.model.predict(X_reshaped, verbose=0)
            return np.argmax(y_pred_proba, axis=1)
        
        def predict_proba(self, X):
            X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
            return self.model.predict(X_reshaped, verbose=0)
    
    # Get number of features and classes
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    
    return CNNClassifier(input_shape=(num_features, 1), num_classes=num_classes)


def train_evaluate_random_forest(X, y):
    """
    Random Forest implementation based on paper specifications:
    - 500 trees
    - Max depth of 30
    - Bootstrap aggregation
    - Class weight balancing
    - Min samples per leaf = 5
    """
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_leaf=5,
        class_weight='balanced',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    return rf_model

def train_evaluate_svm(X, y):
    """
    SVM implementation based on paper specifications:
    - RBF kernel
    - C = 10
    - gamma = 0.001
    - One-vs-rest strategy
    - Class weight balancing
    """
    svm_model = SVC(
        kernel='rbf',
        C=10,
        gamma=0.001,
        decision_function_shape='ovr',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    return svm_model

def train_evaluate_xgboost(X, y):
    """
    XGBoost implementation based on paper specifications:
    - 1000 boosting rounds
    - Max depth of 6
    - Learning rate of 0.01
    - L1 regularization (alpha) = 0.01
    - L2 regularization (lambda) = 1.0
    - Feature sampling = 0.8
    """
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric=['mlogloss', 'merror'],
        use_label_encoder=False
    )
    return xgb_model


def calculate_specificity(y_true, y_pred, class_labels):
    """
    Calculate specificity for each class using one-vs-rest approach
    
    Specificity = TN / (TN + FP)
    """
    specificities = []
    for label in range(len(class_labels)):
        # Convert to binary problem
        y_true_binary = (y_true == label).astype(int)
        y_pred_binary = (y_pred == label).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        # Calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
    
    return np.array(specificities)


def calculate_metrics(y_true, y_pred, y_pred_proba, class_labels=None):
    """
    Calculate all metrics mentioned in the paper, both overall and per-class
    """
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    
    # Calculate specificity
    specificities = calculate_specificity(y_true, y_pred, class_labels)
    
    # Overall metrics
    overall_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'auc_roc': roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr'),
        'specificity': np.mean(specificities)  # Average specificity across all classes
    }
    
    # Per-class metrics
    per_class_metrics = {
        'precision_per_class': precision_score(y_true, y_pred, average=None),
        'recall_per_class': recall_score(y_true, y_pred, average=None),
        'f1_per_class': f1_score(y_true, y_pred, average=None),
        'auc_roc_per_class': roc_auc_score(y_true_bin, y_pred_proba, average=None),
        'specificity_per_class': specificities
    }
    
    # If class labels are provided, create a dictionary with class names
    if class_labels is not None:
        per_class_metrics = {
            metric_name: {
                class_labels[i]: score for i, score in enumerate(scores)
            }
            for metric_name, scores in per_class_metrics.items()
        }
    
    return {
        'overall': overall_metrics,
        'per_class': per_class_metrics
    }


def cross_validate_model(model_func, X, y, model_name, class_labels=None):
    """Perform 5-fold cross-validation and calculate metrics"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    print(f"\nTraining {model_name}...")
    print("-" * 50)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Set random seed for each fold to ensure reproducibility
        set_global_random_seed(42 + fold)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Initialize and train model
        model = model_func(X_train, y_train)
        
        # Special handling for XGBoost: use validation set for early stopping
        if isinstance(model, xgb.XGBClassifier):
            # Further split training data for early stopping in XGBoost
            train_size = 0.8
            from sklearn.model_selection import train_test_split
            X_train_sub, X_eval, y_train_sub, y_eval = train_test_split(
                X_train, y_train, 
                train_size=train_size,
                stratify=y_train,
                random_state=42
            )
            model.fit(
                X_train_sub, y_train_sub,
                eval_set=[(X_eval, y_eval)],
                verbose=0
            )
        else:
            model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)
        
        # Calculate metrics
        fold_result = calculate_metrics(y_val, y_pred, y_pred_proba, class_labels)
        fold_metrics.append(fold_result)
        
        print(f"Fold {fold} Results:")
        for metric, value in fold_result['overall'].items():
            print(f"{metric}: {value:.3f}")
        print("-" * 30)
    
    # Calculate mean and std of metrics
    # For overall metrics
    mean_metrics = {
        'overall': {
            metric: np.mean([fold['overall'][metric] for fold in fold_metrics])
            for metric in fold_metrics[0]['overall'].keys()
        }
    }
    
    std_metrics = {
        'overall': {
            metric: np.std([fold['overall'][metric] for fold in fold_metrics])
            for metric in fold_metrics[0]['overall'].keys()
        }
    }
    
    # For per-class metrics
    if class_labels:
        mean_metrics['per_class'] = {}
        std_metrics['per_class'] = {}
        
        for metric in fold_metrics[0]['per_class'].keys():
            mean_metrics['per_class'][metric] = {}
            std_metrics['per_class'][metric] = {}
            
            for class_name in class_labels:
                values = [fold['per_class'][metric][class_name] for fold in fold_metrics]
                mean_metrics['per_class'][metric][class_name] = np.mean(values)
                std_metrics['per_class'][metric][class_name] = np.std(values)
    
    return {
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'fold_metrics': fold_metrics
    }

def save_results(results, model_name):
    """Save results to JSON file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('outputs/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'{model_name}_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {output_file}")


def print_final_results(results, model_name, class_labels=None):
    """
    Print final results in the format matching both tables from the paper
    """
    mean_metrics = results['mean_metrics']
    std_metrics = results['std_metrics']
    
    # Print overall results (Table 1)
    print(f"\nFinal {model_name} Overall Results (Table 1):")
    print("-" * 50)
    print(f"Accuracy: {mean_metrics['overall']['accuracy']*100:.1f}% ± {std_metrics['overall']['accuracy']*100:.1f}%")
    print(f"F1-Score: {mean_metrics['overall']['f1_score']:.2f} ± {std_metrics['overall']['f1_score']:.2f}")
    print(f"AUC-ROC: {mean_metrics['overall']['auc_roc']:.2f} ± {std_metrics['overall']['auc_roc']:.2f}")
    print(f"Precision: {mean_metrics['overall']['precision']:.2f} ± {std_metrics['overall']['precision']:.2f}")
    print(f"Recall: {mean_metrics['overall']['recall']:.2f} ± {std_metrics['overall']['recall']:.2f}")
    print(f"Specificity: {mean_metrics['overall']['specificity']:.2f} ± {std_metrics['overall']['specificity']:.2f}")
    
    # Print per-class results (Table 2)
    if class_labels:
        print(f"\nDisease-Specific Performance Analysis (Table 2):")
        print("-" * 80)
        print("Disease Condition  Precision  Recall    F1-Score  AUC-ROC   Specificity")
        print("-" * 80)
        
        for class_name in class_labels:
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
            
            print(f"{class_name:<16} {precision:.2f}±{precision_std:.2f}  "
                  f"{recall:.2f}±{recall_std:.2f}  "
                  f"{f1:.2f}±{f1_std:.2f}  "
                  f"{auc:.2f}±{auc_std:.2f}  "
                  f"{specificity:.2f}±{specificity_std:.2f}")

def train_all_models(X, y, class_labels=None):
    """Train and evaluate all models with per-class metrics"""
    models = {
        'Random Forest': train_evaluate_random_forest,
        'SVM': train_evaluate_svm,
        'XGBoost': train_evaluate_xgboost,
        'CNN': train_evaluate_cnn,
        'Transformer': train_evaluate_transformer
    }
    
    all_results = {}
    
    for model_name, model_func in models.items():
        results = cross_validate_model(model_func, X, y, model_name, class_labels)
        all_results[model_name] = results
        
        print_final_results(results, model_name)
        save_results(results, model_name.lower().replace(' ', '_'))
    
    return all_results