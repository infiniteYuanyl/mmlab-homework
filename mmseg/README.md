# mmseg作业说明

### 1.基础作业

**数据集说明：**

<img src="https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230210155453678.png" alt="image-20230210155453678" style="zoom:50%;" />使用Kaggle中People Clothing数据集

数据集共1000张图片，划分为800张训练集和200张验证集。

使用`deeplabv3+`网络进行训练160k个iteration

训练16k `iteration`后，可视化并处理效果如下

<center>
    <img src="https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230211010438582.png" alt="image-20230211010438582" style="zoom: 25%;" />
    <img src="https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230211010356939.png" alt="image-20230211010356939" style="zoom: 25%;" />
    <img src="https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230211012449569.png" alt="image-20230211012449569" style="zoom:25%;" />
    <img src="https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230211012331910.png" alt="image-20230211012331910" style="zoom:25%;" /><center/>



貌似皮肤skin和裙子没有分辨出来，还是有待继续训练。

日志仍然在base的日期目录下，配置文件在`base`目录下

`16k` iter训练模型文件链接：

链接：https://pan.baidu.com/s/165CMeclmeU5bcjbv8U4Qpw 
提取码：8zee 

### 2.进阶作业

**数据集说明：**

使用vaihingen公开遥感数据集，其中使用`tools/dataset_converters/vaihingen`工具进行对数据的处理，主要分为两个方面：1.遥感图像由于分辨率非常高，达到`6000 * 6000`，因此需要对每个图像进行分割成合适的输入图片2.将图片划分为`600`张左右的训练集和`150`张左右的验证集。

**模型：**

使用`mask2former`模型进行分割，其最大的特点是，对于使用`swin`作为`backbone`的模型来说，batchsize需要设置为2才能够放24g显存的gpu上（一张图片10多个G，属实是大力出奇迹了）。



**模型配置文件在`advance`的目录下，日志文件在advance的日期目录下。模型训练文件放在最后百度网盘中**

训练模型文件链接：

链接：https://pan.baidu.com/s/1DhyMTqXIsS-zIgHC3HwJuw 提取码：a12r 

**训练过程&结果：**

![image-20230210153401286](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230210153401286.png)

训练了90k的`iteration`，可见没有完全的收敛，还可以继续训练，不过这里仅作为作业来说，进度也非常够用了。

最终在验证集上可以达到`mIou`=75.97，`mAcc`=85.52的指标，可以说是相当的高了。

**效果展示**

左边为真值（mask）,右边为预测值，可见其效果是不错的，但仍有提升空间。

![image-20230210153851884](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230210153851884.png)






