# Hawk-I

基于无监督机器学习算法从Web日志中自动提取攻击Payload。

### 思路

要把异常参数找出来，最显而易见要解决的问题就是如何量化请求中各参数的异常程度。为了最大化利用日志中蕴含的需要保护的Web系统自身的结构信息，我决定对请求按访问路径进行拆解，即分析**参数value在同路径同参数Key的其他参数值中的异常程度**。具体算法步骤是：

1）	基于TF-IDF对不同路径下的样本分别进行特征向量化，按参数维度对特征向量进行**汇聚**。

2）	基于特征向量提取出样本参数在同路径同参数Key的其他参数值中异常分数AS（Anomaly Score）。

3） 设置阈值T，取出AS大于T的异常参数值作为输出。

### 数据集
使用HTTP CSIC 2010数据集。当然使用任何其他数据集也可。但注意根据数据集格式调整解析逻辑data/parse.py。

### 用法
    #将原始数据集解析成JSON格式数据集。
    python data/parse.py
    
    #向量化
    python vectorize/vectorizer.py
    
    #通过学习曲线评估特征向量的有效性
    python evaluate/learningCurve.py
    
    #提取异常Payload
    python score/as.py
    
本项目不断完善中，且后面会增加访问时序的异常性评估，从而结合Web系统结构信息，更精确地识别异常访问。