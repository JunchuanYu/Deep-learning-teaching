# **深度学习入门培训**

## 作者：于峻川
## 联系方式：yujunchuan@mail.cgs.gov.cn
<br/>

## **1\-学习资源**

* 动手学深度学习（李沐）https://courses.d2l.ai/zh-v2/
* kaggle网站：https://www.kaggle.com/learn/computer-vision
* Tensorflow网站：https://tensorflow.google.cn/tutorials?hl=zh_cn
* Keras网站：https://keras.io/
* Keas视频教程：https://www.bilibili.com/video/BV1TW411Y7HU?spm_id_from=333.999.0.0
* Pytorch视频教程：https://www.bilibili.com/video/BV1Zv4y1o7uG?share_source=copy_web
* [其他资源](https://blog.csdn.net/Datawhale/article/details/104509347)

## **2\-课程安排**

1. **环境配置与安装说明**
    * 安装anaconda3配置keras环境，conda create \-n kears gdal=3 python=3.6
    * 根据显卡型号配备的cuda和cudnn, 20系和tesla系显卡cuda10即可，因此可同时配置tf1及tf2,30系显卡需要cuda11，必须配置tf2
    * 安装tensorflow\-gpu,keras,gdal,pandas,opencv\-python,sklearn,scikte\-image,matplotlib,Image,pillow,jupyterr等库, pip instal ×××
    * 安装vscode，其他可选（jupyter notebook,spyder,pycharm），重装需要删除C:\\Users\\jason\\AppData\\Roaming\\Code和C:\\Users\\jason\\.vscode
    * tf2整合了keras注意框架的差异[tf.keras和Keras区别](https://blog.csdn.net/weixin_40920183/article/details/106718229)
2. **深度学习概述**
    * 深度学习的进展
        * 深度学习是热点，面向对象是主流，人工解译是常态
        * 网络框架与模型不断迭代
        * 深入行业占领易达高地向深部进军
        * 计算机视觉牵引应用行业发展
    * 深度学习的困境
        * 数据匮乏小样本问题
        * 数据标记难度大
        * 业务场景结合专家知识模型化
        * 模型过拟合与泛化
        * 模型优化与算力
        * 可解释性不足
        * 缺少成熟商用软件系统
    * 深度学习的发展方向
        * 由监督\-非监督发展（自编码）
        * 由大样本\-小样本发展（gan，场景模拟）
        * 由大模型\-小模型发展（轻量化设计\-剪枝）
        * 由数据驱动\-知识与数据驱动融合发展（知识模型化）
        * 由黑盒\-灰盒发展（可解释性研究）
        * 由小作坊\-工程化发展（云计算）
    * 深度学习的学习思路
        * 机器学习基础知识
        * python基础操作
        * tf，keras，pytorch框架学习
        * 经典网络结构单元学习
        * 经典网络复现
        * 大量数据集和消融实验实战
        * 网络设计与效率精度提升
        * 根据应用需求开展工程应用
3. **深度学习实战**
    * 案例1\-手写字体识别：熟悉keras框架
        * 数据加载及标准化处理
        * 函数式和贯通式模型构建
        * 卷积的维度
        * 模型训练及预测
        * 模型保存和加载
        * 练习任务1：构建更深的网络训练更好的模型
        * 练习任务2：学习纯numpy构建简单的多层感知机实现分类
        * 练习任务3：用二维卷积构建网络实现对手写图片的分类
    * 案例2\-海冰分割：构建unet网络
        * png格式图片数据读取及预处理
        * unet模型构建
        * 模型超参数设置
        * 训练数据输出与成图
        * 练习任务1：运用ImageDataGenerator对数据进行增广
        * 练习任务2: 构建自定义generator分别实现对数据实体和数组的分批读取
        * 练习任务3：优化unet模型
    * 案例3\-遥感地物识别：熟悉deeplabV3及精度评价
        * [数据下载链接](https://pan.baidu.com/s/1dHK5-5-tMGPeSCGJqR0YqA) 提取码：1111
        * 熟悉深度学习模型库的架构
        * 熟悉遥感数据切片方案
        * 熟悉deeplab V3网络
        * 熟悉分类精度评价方法
        * 练习任务1: 替换不同的backbone进行消融实验
        * 练习任务2: 改进decoder重构策略

    * 案例4\-高光谱分类实验：构建CbrrUnet网络完成地物识别和评价
        * 数据下载链接：请联系作者(yujunchuan@mail.cgs.gov.cn)
        * 了解遥感数据读写方法
        * 半自动交互式样本集构建
        * 高光谱处理技术
        * 大区域目标推理策略
        * 练习任务1: 与unet模型开展对比试验
        * 练习任务2:  通过修改loss提升多分类性能
