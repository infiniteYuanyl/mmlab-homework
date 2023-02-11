# mmseg作业说明

### 1.基础作业

**数据集说明：**

<img src="https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230210155453678.png" alt="image-20230210155453678" style="zoom:50%;" />使用Kaggle中People Clothing数据集

数据集共1000张图片，划分为800张训练集和200张验证集。

使用`deeplabv3+`网络进行训练160k个iteration

训练16k `iteration`后，可视化并处理效果如下

![image-20230211013951573](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230211013951573.png)

貌似皮肤skin和裙子没有分辨出来，还是有待继续训练。

日志仍然在base的日期目录下，配置文件在`base`目录下.

以下是各类的指标，可以看到一些类预测几乎为0，可能是因为验证集压根没有这个类别导致的。由于超算平台没钱了的关系，在有限的时间内完成度也就这样了。

02/11 20:12:26 - mmengine - INFO - 
+-------------+-------+-------+
|    Class    |  IoU  |  Acc  |
+-------------+-------+-------+
|  background | 96.51 | 98.58 |
| accessories |  0.12 |  0.13 |
|     bag     | 32.87 | 51.96 |
|     belt    |  9.48 | 11.33 |
|    blazer   |  3.75 |  5.13 |
|    blouse   |  10.5 |  22.1 |
|   bodysuit  |  0.0  |  0.0  |
|    boots    |  0.0  |  0.0  |
|     bra     |  0.0  |  0.0  |
|   bracelet  |  0.02 |  0.02 |
|     cape    |  0.03 |  0.05 |
|   cardigan  |  0.06 |  0.07 |
|    clogs    |  0.0  |  0.0  |
|     coat    | 32.67 | 43.35 |
|    dress    | 41.55 | 71.35 |
|   earrings  |  0.0  |  0.0  |
|    flats    |  0.0  |  0.0  |
|   glasses   |  0.0  |  0.0  |
|    gloves   |  0.0  |  0.0  |
|     hair    | 59.48 | 90.99 |
|     hat     | 18.09 | 18.88 |
|    heels    |  0.0  |  0.0  |
|    hoodie   |  0.0  |  0.0  |
|   intimate  |  0.0  |  0.0  |
|    jacket   | 28.13 | 51.57 |
|    jeans    |  30.0 | 44.56 |
|    jumper   |  0.0  |  0.0  |
|   leggings  |  6.06 |  7.34 |
|   loafers   |  0.06 |  0.06 |
|   necklace  |  0.03 |  0.03 |
|   panties   |  0.0  |  0.0  |
|    pants    | 44.52 | 68.65 |
|    pumps    |  0.0  |  0.0  |
|    purse    |  2.77 |  7.07 |
|     ring    |  0.0  |  0.0  |
|    romper   |  0.0  |  0.0  |
|   sandals   |  1.02 |  1.08 |
|    scarf    |  1.84 |  2.07 |
|    shirt    | 16.44 | 43.55 |
|    shoes    |  45.8 | 67.31 |
|    shorts   | 10.83 |  14.1 |
|     skin    | 67.28 | 93.83 |
|    skirt    |  7.2  |  8.38 |
|   sneakers  |  0.54 |  0.55 |
|    socks    | 10.08 | 11.05 |
|  stockings  | 27.06 | 44.77 |
|     suit    | 28.35 | 63.42 |
|  sunglasses | 29.59 | 35.97 |
|   sweater   | 22.92 | 37.14 |
|  sweatshirt |  0.26 |  0.3  |
|   swimwear  |  0.0  |  0.0  |
|   t-shirt   | 17.99 | 22.39 |
|     tie     |  7.42 |  7.75 |
|    tights   |  0.15 |  0.17 |
|     top     |  5.14 |  5.66 |
|     vest    |  0.02 |  0.02 |
|    wallet   |  0.0  |  0.0  |
|    watch    |  0.0  |  0.0  |
|    wedges   |  0.0  |  0.0  |
+-------------+-------+-------+

` aAcc`: 86.3400  `mIoU`: 12.1500  `mAcc`: 17.8400

`16k` iter训练模型文件链接：

链接：https://pan.baidu.com/s/165CMeclmeU5bcjbv8U4Qpw 
提取码：8zee 

### 2.进阶作业

**数据集说明：**

使用vaihingen公开遥感数据集，其中使用`tools/dataset_converters/vaihingen`工具进行对数据的处理，主要分为两个方面：1.遥感图像由于分辨率非常高，达到`6000 * 6000`，因此需要对每个图像进行分割成合适的输入图片2.将图片划分为`600`张左右的训练集和`150`张左右的验证集。

**模型：**

使用`mask2former`模型进行分割，其最大的特点是，对于使用`swin`作为`backbone`的模型来说，batchsize需要设置为2才能够放24g显存的gpu上（一张图片10多个G，属实是大力出奇迹了）。由于没设备了，我还是选择`resnet50`作为`backbone`



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






