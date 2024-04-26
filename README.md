# 关键点检测算法DOG

### 图像特征提取之DoG算子

（1）读取原始图像img

（2）对原始图像img在相同核大小，不同标准差下做高斯滤波，例如分别为sigma1，sigma2,sigma3,sigma4做高斯滤波得到图像G1，G2，G3，G4；

（3）对相邻的滤波后的图像做差值，得到差值图像D1=G2-G1，D2=G3-G2，D3=G4-G3

（4）对于中间的一副差值图像D2，遍历图中所有的点，对于每一点D2（i，j）做如下的操作

（4-1）在图D2中这个点D2（i，j）一定的领域内，例如3*3的范围内，以及D1和D3中这个点D2（i，j）对应的相同位置处的点D1（i，j）和 D3（i，j）的相同领域内，求点D2（i，j）是否为极值点（极大值或极小值），如果为极值点，则标记为角点。

![image-20240424145107364](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145107364.png)

# 尺度变化不变特征SIFT

### 特征匹配

![image-20240424145325716](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145325716.png)![image-20240424145906379](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145906379.png)



![image-20240424145405493](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145405493.png)![image-20240424145921463](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145921463.png)![image-20240424145936147](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145936147.png)



# 图像拼接

### 两张图像

![image-20240424145550978](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145550978.png)

![image-20240424145605817](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145605817.png)![image-20240424145622635](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145622635.png)

![image-20240424145634725](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145634725.png)

![image-20240424145646979](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145646979.png)

### 三张图像

![image-20240424145721787](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145721787.png)

![image-20240424145736332](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145736332.png)

![image-20240424145752550](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145752550.png)![image-20240424145804821](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145804821.png)![image-20240424145819928](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/blob/main/results/image-20240424145819928.png)