外部数据
使用 LUng Node Analysis Grand Challenge 数据，因为这个数据集包含了来自放射学的标注细节。
使用 LIDC-IDRI 数据，因为它具有找到了肿瘤的所有放射学的描述。
使用Flickr CC，维基百科通用数据集
使用Human Protein Atlas Dataset
使用IDRiD数据集
数据探索和直觉
使用0.5的阈值对3D分割进行聚类
确认在训练集和测试集的标签分布上有没有不一样的地方

预处理
使用DoG（Difference of Gaussian）方法进行blob检测，使用skimage中的方法。
使用基于patch的输入进行训练，为了减少训练时间。
使用cudf加载数据，不要用Pandas，因为读数据更快。
确保所有的图像具有相同的方向。
在进行直方图均衡化的时候，使用对比度限制。
使用OpenCV进行通用的图像预处理。
使用自动化主动学习，添加手工标注。
将所有的图像缩放成相同的分辨率，可以使用相同的模型来扫描不同的厚度。
将扫描图像归一化为3D的numpy数组。
对单张图像使用暗通道先验方法进行图像去雾。
将所有图像转化成Hounsfield单位（放射学中的概念）。
使用RGBY的匹配系数来找到冗余的图像。
开发一个采样器，让标签更加的均衡。
对测试图像打伪标签来提升分数。
将图像/Mask降采样到320x480。
直方图均衡化（CLAHE）的时候使用kernel size为32×32
将DCM转化为PNG。
当有冗余图像的时候，为每个图像计算md5 hash值。
数据增强
使用 albumentations 进行数据增强。
使用随机90度旋转。
使用水平翻转，上下翻转。
可以尝试较大的几何变换：弹性变换，仿射变换，样条仿射变换，枕形畸变。
使用随机HSV。
使用loss-less增强来进行泛化，防止有用的图像信息出现大的loss。
应用channel shuffling。
基于类别的频率进行数据增强。
使用高斯噪声。
对3D图像使用lossless重排来进行数据增强。
0到45度随机旋转。
从0.8到1.2随机缩放。
亮度变换。
随机变化hue和饱和度。
使用D4：https://en.wikipedia.org/wiki/Dihedral_group增强。
在进行直方图均衡化的时候使用对比度限制。
使用AutoAugment：https://arxiv.org/pdf/1805.09501.pdf增强策略。
模型
结构
使用U-net作为基础结构，并调整以适应3D的输入。
使用自动化主动学习并添加人工标注。
使用inception-ResNet v2 architecture结构使用不同的感受野训练特征。
使用Siamese networks进行对抗训练。
使用ResNet50, Xception, Inception ResNetv2 x 5，最后一层用全连接。
使用global max-pooling layer，无论什么输入尺寸，返回固定长度的输出。
使用stacked dilated convolutions。
VoxelNet。
在LinkNet的跳跃连接中将相加替换为拼接和conv1x1。
Generalized mean pooling。
使用224x224x3的输入，用Keras NASNetLarge从头训练模型。
使用3D卷积网络。
使用ResNet152作为预训练的特征提取器。
将ResNet的最后的全连接层替换为3个使用dropout的全连接层。
在decoder中使用转置卷积。
使用VGG作为基础结构。
使用C3D网络，使用adjusted receptive fields，在网络的最后使用64 unit bottleneck layer 。
使用带预训练权重的UNet类型的结构在8bit RGB输入图像上提升收敛性和二元分割的性能。
使用LinkNet，因为又快又省内存。
MASKRCNN
BN-Inception
Fast Point R-CNN
Seresnext
UNet and Deeplabv3
Faster RCNN
SENet154
ResNet152
NASNet-A-Large
EfficientNetB4
ResNet101
GAPNet
PNASNet-5-Large
Densenet121
AC-GAN
XceptionNet (96), XceptionNet (299), Inception v3 (139), InceptionResNet v2 (299), DenseNet121 (224)
AlbuNet (resnet34) from ternausnets
SpaceNet
Resnet50 from selim_sef SpaceNet 4
SCSEUnet (seresnext50) from selim_sef SpaceNet 4
A custom Unet and Linknet architecture
FPNetResNet50 (5 folds)
FPNetResNet101 (5 folds)
FPNetResNet101 (7 folds with different seeds)
PANetDilatedResNet34 (4 folds)
PANetResNet50 (4 folds)
EMANetResNet101 (2 folds)
RetinaNet
Deformable R-FCN
Deformable Relation Networks
硬件设置
Use of the AWS GPU instance p2.xlarge with a NVIDIA K80 GPU
Pascal Titan-X GPU
Use of 8 TITAN X GPUs
6 GPUs: 21080Ti + 41080
Server with 8×NVIDIA Tesla P40, 256 GB RAM and 28 CPU cores
Intel Core i7 5930k, 2×1080, 64 GB of RAM, 2x512GB SSD, 3TB HDD
GCP 1x P100, 8x CPU, 15 GB RAM, SSD or 2x P100, 16x CPU, 30 GB RAM
NVIDIA Tesla P100 GPU with 16GB of RAM
Intel Core i7 5930k, 2×1080, 64 GB of RAM, 2x512GB SSD, 3TB HDD
980Ti GPU, 2600k CPU, and 14GB RAM
损失函数
Dice Coefficient ，因为在不均衡数据上工作很好。
Weighted boundary loss 目的是减少预测的分割和ground truth之间的距离。
MultiLabelSoftMarginLoss 使用one-versus-all损失优化多标签。
Balanced cross entropy (BCE) with logit loss 通过系数来分配正负样本的权重。
Lovasz 基于sub-modular损失的convex Lovasz扩展来直接优化平均IoU损失。
FocalLoss + Lovasz 将Focal loss和Lovasz losses相加得到。
Arc margin loss 通过添加margin来最大化人脸类别的可分性。
Npairs loss 计算y_true 和 y_pred之间的npairs损失。
将BCE和Dice loss组合起来。
LSEP – 一种成对的排序损失，处处平滑因此容易优化。
Center loss 同时学习每个类别的特征中心，并对距离特征中心距离太远的样本进行惩罚。
Ring Loss 对标准的损失函数进行了增强，如Softmax。
Hard triplet loss 训练网络进行特征嵌入，最大化不同类别之间的特征的距离。
1 + BCE – Dice 包含了BCE和DICE损失再加1。
Binary cross-entropy –  log(dice) 二元交叉熵减去dice loss的log。
BCE, dice和focal 损失的组合。
BCE + DICE - Dice损失通过计算平滑的dice系数得到。
Focal loss with Gamma 2 标准交叉熵损失的升级。
BCE + DICE + Focal – 3种损失相加。
Active Contour Loss 加入了面积和尺寸信息，并集成到深度学习模型中。
1024 * BCE(results, masks) + BCE(cls, cls_target)
Focal + kappa – Kappa是一种用于多类别分类的损失，这里和Focal loss相加。
ArcFaceLoss —  用于人脸识别的Additive Angular Margin Loss。
soft Dice trained on positives only – 使用预测概率的Soft Dice。
2.7 * BCE(pred_mask, gt_mask) + 0.9 * DICE(pred_mask, gt_mask) + 0.1 * BCE(pred_empty, gt_empty) 一种自定义损失。
nn.SmoothL1Loss()。
使用Mean Squared Error objective function，在某些场景下比二元交叉熵损失好。
训练技巧
尝试不同的学习率。
尝试不同的batch size。
使用SGD + 动量 并手工设计学习率策略。
太多的增强会降低准确率。
在图像上进行裁剪做训练，全尺寸图像做预测。
使用Keras的ReduceLROnPlateau()作为学习率策略。
不使用数据增强训练到平台期，然后对一些epochs使用软硬增强。
冻结除了最后一层外的所有层，使用1000张图像进行微调，作为第一步。
使用分类别采样
在调试最后一层的时候使用dropout和增强
使用伪标签来提高分数
使用Adam在plateau的时候衰减学习率
用SGD使用Cyclic学习率策略
如果验证损失持续2个epochs没有降低，将学习率进行衰减
将10个batches里的最差的batch进行重复训练
使用默认的UNET进行训练
对patch进行重叠，这样边缘像素被覆盖两次
超参数调试：训练时候的学习率，非极大值抑制以及推理时候的分数阈值
将低置信度得分的包围框去掉。
训练不同的卷积网络进行模型集成。
在F1score开始下降的时候就停止训练。
使用不同的学习率。
使用层叠的方法用5 folds的方法训练ANN，重复30次。
评估和验证
按类别非均匀的划分训练和测试集
当调试最后一层的时候，使用交叉验证来避免过拟合。
使用10折交叉验证集成来进行分类。
检测的时候使用5-10折交叉验证来集成。
集成方法
使用简单的投票方法进行集成
对于类别很多的模型使用LightGBM，使用原始特征。
对2层模型使用CatBoost。
使用 ‘curriculum learning’ 来加速模型训练，这种训练模式下，模型先在简单样本上训练，再在困难样本上训练。
使用ResNet50, InceptionV3, and InceptionResNetV2进行集成。
对物体检测使用集成。
对Mask RCNN, YOLOv3, 和Faster RCNN 进行集成。
后处理
使用test time augmentation ，对一张图像进行随机变换多次测试后对结果进行平均。
对测试的预测概率进行均衡化，而不是使用预测的类别。
对预测结果进行几何平均。
在推理的时候分块重叠，因为UNet对边缘区域的预测不是很好。
进行非极大值抑制和包围框的收缩。
在实例分割中使用分水岭算法后处理来分离物体。
