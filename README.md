# Multi-stream Adaptive Feature Integration for Microbiome Disease Classification

本项目实现了基于多流自适应特征集成的微生物组疾病分类方法。

## 运行对比实验

### 环境要求
- Python 3.12+
- 相关依赖包（建议使用 conda 或 venv 创建虚拟环境）

### 安装依赖
```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖包
pip install -r requirements.txt
```

### 运行对比实验
1. 确保数据集已放置在 `data/` 目录下
2. 执行以下命令运行对比实验：
```bash
python comparative_experiment.py
```

### 实验输出
- 实验结果将保存在 `outputs/` 目录下
- 包含各个对比模型的性能指标（准确率、精确率、召回率、F1分数等）
- 可视化结果（如果有）也会保存在该目录

### 注意事项
- 请确保有足够的磁盘空间用于存储实验结果
- 大规模数据集的实验可能需要较长时间，请耐心等待
- 如遇到内存不足问题，可以适当调整批处理大小

## 项目结构
```
├── README.md               # 项目说明文档
├── comparative_experiment.py    # 对比实验主程序
├── ml_models.py           # 机器学习模型定义
├── data/                  # 数据集目录
├── outputs/               # 输出结果目录
└── __pycache__/          # Python 缓存文件
```

## 问题排查
如果遇到运行问题，请检查：
1. Python 环境版本是否正确
2. 所有依赖包是否正确安装
3. 数据集格式是否符合要求
4. 输出目录是否具有写入权限

## 引用
如果您使用了本项目的代码，请引用：
[论文引用信息待补充]