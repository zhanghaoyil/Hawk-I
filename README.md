# Hawk-I
<<<<<<< HEAD
Automatic extract anomalious Web attack Payloads with Unsupervised Machine Learning algorithms
=======
Automatic extract anomalious Web attack Payloads with Unsupervised Machine Learning algorithms.

### 思路
要把异常参数找出来，最显而易见要解决的问题就是如何量化请求中各参数的异常程度。本文的思路相当朴素，比朴素贝叶斯还朴素得多——参数的异常程度取决于其所在请求在同路径的其他请求中的异常程度，以及参数值在同路径同参数Key的其他参数值中的异常程度。具体算法步骤是：

1）	基于TF-IDF对不同路径下的样本分别进行特征向量化。

2）	运用无监督学习算法对样本在同路径下所有其他请求中的异常程度进行评估，获取样本异常分数SAS（Sample Anomaly Score）。

3）	基于TF-IDF向量提取出样本参数在同路径同参数Key的其他参数值中异常分数PAS（Param Anomaly Score）。

4）	计算所有样本中所有参数的异常分数AS = SAS * PAS。

5）	设置阈值T，取出所有AS > T的参数值作为输出。

### 数据集
使用HTTP CSIC 2010数据集。当然使用任何其他数据集也可。但注意根据数据集格式调整解析逻辑data/parse.py

### 用法
    #将原始数据集解析成JSON格式数据集。
    python data/parse.py
    
    #向量化
    python vectorize/vectorizer.py
    
    #
