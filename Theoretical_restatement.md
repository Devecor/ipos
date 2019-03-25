# 理论重述

### 1. 概述

本文明确使用单目视觉, 针孔相机模型<sup>[1]</sup>, orb特征检测与匹配<sup>[2]</sup>
相关理论: 对极几何, 三角测量, PnP


### 2. 对极几何

#### 2.1 坐标系

* __像素坐标系:__ 
图像的左上角是原点， 正方向分别往右和往下
* __相机坐标系:__
原点为相机镜头的光心， 向前为z轴正方向， 向下为y轴正方向， 向右为x轴正方向
* __归一化坐标:__
相机坐标系投影到z=1的平面上的坐标， 显然， 坐标形如(X/Z, Y/Z, 1)的形式
* __世界坐标系:__
可以任意指定， 为方便起见， 本文取世界坐标为右侧相机的相机坐标系为世界坐标系

#### 2.2 极线约束

以第一帧图像的相机坐标系为世界坐标系, 某一空间点P = [X, Y, Z]<sup>T</sup>, 投影在两张图像的点像素坐标分别为$p_1,p_2$ 以$s$表示深度，则有
$$
\begin{cases}
s_1p_1=KP\\\\
s_2p_2=K(RP+t)
\end{cases}
$$
简化为以下形式，其中$E=t^{\land}R$，称为本征矩阵
<center>[p2;1]<sup>T</sup>K<sup>-T</sup>EK<sup>-1</sup>[p1;1] = 0</center>

通过E可以分解出R和t, 本身具有尺度等价性, 但是对t乘以一个非0常数, 分解也是成立的<sup>[3]</sup>，**所以通常对t进行归一化处理，使其长度为1，这将直接导致t的尺度不确定性**，比如t的第一维是0.17, 我们无法确定其单位是cm还是m. 因此单目slam有一步不可避免的初始化, 初始化的图像必须有一定程度的平移, 而后的轨迹和地图必须以此步的平移为单位.

### 3. 三维重建

根据针孔相机模型， 有如下关系: 
$$
Z\begin{bmatrix}
u\\\\
v\\\\
1
\end{bmatrix} = 
\begin{bmatrix}
f_x&0&c_x\\\\
0&f_y&c_y\\\\
0&0&1\end{bmatrix}
\begin{bmatrix}
X\\\\
Y\\\\
Z\end{bmatrix}
$$
说明：$[u\ v\ 1]^T$ 是像素坐标，$[X\ Y\ Z]^T$是某一点的空间坐标。

仅通过上式无法求解空间坐标，但是能求解空间点映射到归一化平面$(Z=1)$上的归一化坐标$\begin{bmatrix}u'&v'&1\end{bmatrix}$，或称为像素坐标在归一化平面上的齐次坐标，显然
$$
\begin{bmatrix}u'&v'&1\end{bmatrix}^T=
\begin{bmatrix}X/Z&Y/Z&1\end{bmatrix}^T
$$
使用三角测量求解深度，设某一空间点在两帧图像上的深度为$s_1和s_2$，以第一帧图像的相机坐标系为世界坐标系，根据对极几何，
$$
s_1\begin{bmatrix}u_1\\\\v_1\\\\1\end{bmatrix} =
s_2\begin{bmatrix}
r_1&r_2&r_3\\\\
r_4&r_5&r_6\\\\
r_7&r_8&r_9
\end{bmatrix}
\begin{bmatrix}u_2\\\\v_2\\\\1\end{bmatrix} +
\begin{bmatrix}t_1\\\\t_2\\\\t_3\end{bmatrix}
$$
记为
$$
s_1x_1=s_2Rx_2 + t
$$
其中$R\ $表示图像的相对旋转，$t\ $表示相对平移

在本文设定的坐标系下，$s_1=Z$，对上式两边左乘$x^{\land}_1$，可求得$s_2$，进一步求得$s_1$

上式所用$R和t$，一般通过极限约束求得本征矩阵$E$，再分解本征矩阵得到$R和t$，本实验的两张参考照片的相对变换仅在$x$轴方向平移了`10cm`，__则旋转变换矩阵$R$必然是3阶单位阵，平移变换矩阵为$\begin{bmatrix}10&0&0\end{bmatrix}^T$__，不必再使用opencv求解，这一点从opencv的求解结果中得到了验证。

### 4. 求解PnP，获得参考照片的位姿

#### 4.1 直接线性变换
$$s
\begin{bmatrix}u'\\\\v'\\\\1\end{bmatrix} =
\begin{bmatrix}
r_1&r_2&r_3&t_1\\\\
r_4&r_5&r_6&t_2\\\\
r_7&r_8&r_9&t_3
\end{bmatrix}
\begin{bmatrix}X\\\\Y\\\\Z\\\\1\end{bmatrix}
$$
记为
$$
sx'=\begin{bmatrix}R|t\end{bmatrix}P
$$
这里的$R和t$代表了测试照片相对于参考点的朝向和位置

实际上直接线性变换准确率不高，一般使用其优化算法如EPnP, UPnP等

实验结果表明这种求解相机运动的办法，效果非常差

#### 4.2 坐标变换

<!-- ##### 4.2.1 $R2\theta$ -->

任意两个笛卡尔坐标系之间的旋转变换：
$$
\begin{bmatrix}X'\\\\Y'\\\\Z'\\\\\end{bmatrix}=\begin{bmatrix}
cos\theta_3&sin\theta_3&0\\\\
-sin\theta_3&cos\theta_3&0\\\\
0&0&1
\end{bmatrix}
\begin{bmatrix}
cos\theta_2&0&-sin\theta_2\\\\
0&1&0\\\\
sin\theta_2&0&cos\theta_2
\end{bmatrix}
\begin{bmatrix}
1&0&0\\\\
0&cos\theta_1&sin\theta_1\\\\
0&-sin\theta_1&cos\theta_1
\end{bmatrix}
\begin{bmatrix}X\\\\Y\\\\Z\\\\\end{bmatrix}
$$
说明：这里，我们假设坐标系O-XYZ依次绕自身X轴、Y轴、Z轴分别逆时针转$\theta_1，\theta_2，\theta_3$后可以与坐标系O'-X'Y'Z'重合

使用此方程既可以将R转换为相机的相对角度，也可以将定位结果从相机坐标系变换到世界坐标系




### 5. 参考文献

* [1] 高翔, 张涛. 视觉slam十四讲:从理论到实践[M]. 北京:电子工业出版社. 2017:84-90.
* [2] Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary R. Bradski: ORB: An efficient alternative to SIFT or SURF[C]. ICCV 2011: 2564-2571.
* [3] 高翔, 张涛. 视觉slam十四讲:从理论到实践[M]. 北京:电子工业出版社. 2017:151-152.




