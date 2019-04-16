# PCVch05

记录学习Python Computer Vision的过程

第七次记录

## 三维重建

### 外极几何

外极几何（Epipolar Geometry）描述的是两幅视图之间的内在射影关系，与外部场景无关，只依赖于摄像机内参数和这两幅试图之间的的相对姿态。

![img](https://github.com/zengqq1997/PCVch05-/blob/master/外极几何.jpg)

> 1. 基线（baseline）：直线CC'为基线。
>
> 2. 对极平面束（epipolar pencil）：以基线为轴的平面束。
>
> 3. 对极平面（epipolar plane）：任何包含基线的平面都称为对极平面。
>
> 4. 对极点（epipole）：摄像机的基线与每幅图像的交点。比如，上图中的点e和e'。
>
> 5. 对极线（epipolar line）：对极平面与图像的交线。比如，上图中的直线l和l'。
>
> 6. 5点共面：点x，x'，摄像机中心C、C'，空间点X是5点共面的。
>
> 7. 极线约束：两极线上点的对应关系。
>
>    说明：直线l是对应于点x'的极线，直线l'是对应于点x的极线。极线约束是指点x'一定在对应于x的极线l'上，点x一定在对应于x'的极线l上。

**外极几何实际模型**：一般的立体成像关系是两个相机的坐标无任何约束关系，相机的内部参数可能不同，甚至是未知的。要刻画这种情况下的两幅图像之间的对应关系，需要引入两个重要的概念：本征矩阵E和基本矩阵F。本征矩阵E包含物理空间中两个摄像机相关的旋转和平移信息，基础矩阵F除了包含E的信息外，还包含两个摄像机的内参数。本征矩阵E将左摄像机观测到的点的物理坐标与右摄像机观测到的相同点的位置关联起来。基础矩阵F则是将一台摄像机的像平面上的点在图像坐标（像素）上的坐标和另一台摄像机的像平面上的点关联起来。

### 本征矩阵E（Essential Matrix）

反映空间一点P的像点在不同视角摄像机下摄像机坐标系中的表示之间的关系。

![img](https://github.com/zengqq1997/PCVch05-/blob/master/本质矩阵.jpg)

### 基础矩阵F（Fundamental Matrix）

反映空间一点P的像素点在不同视角摄像机下图像坐标系中的表示之间的关系。

![img](https://github.com/zengqq1997/PCVch05-/blob/master/基础矩阵.jpg)

基本矩阵提供了三维点到二维的一个约束条件。举个例子，现在假设我们不知道空间点X的位置，只知道X在左边图上的投影x的坐标位置，也知道基本矩阵，首先我们知道的是X一定在射线Cx上，到底在哪一点是没法知道的，也就是X可能是Cx上的任意一点(也就是轨迹的意思)，那么X在右图上的投影肯定也是一条直线。也就是说，如果我们知道一幅图像中的某一点和两幅图的基本矩阵，那么就能知道其对应的右图上的点一定是在一条直线上，这样就约束了两视角下的图像中的空间位置一定是有约束的，不是任意的。基本矩阵是很有用的一个工具，在三维重建和特征匹配上都可以用到。

简单的讲基础矩阵表示的是某个物体或场景各特征在不同的两张照片对应特征点图像坐标的关系；对这些图像坐标用照片对应相机内参数进行归一化得到归一化坐标，本质矩阵表示同一特征对应归一化坐标的关系，本质矩阵分解可得到两相机之间旋转矩阵和平移向量。二者联系：利用本质矩阵和相机内参数矩阵相乘可以得到基础矩阵。

**代数推导**：对于两个视图的射影矩阵PP、P′P′，在矩阵PP的作用下，第一个视图中通过xx和光心OO的射线可以由方程PX=xPX=x解出。给出的单参数簇解的形式为： X(λ)=P+x+λO

其中P+是P的伪逆，即P+P=I，O为相机的中心，即P的零矢量并且定义为PO=0。这条射线由点P+x和点O决定，这两点在第二幅图像上的投影分别为点P′P+x和点P′O。而对极线则是连接这两点的直线，即l′=(P′O)×(P′P+x)，点P′O也就是在第二幅图像上的对极点e′。也可以记为l′=[e′]×(P′P+)x=Fx，这里的F就是基本矩阵：F=[e′]×(P′P+)

设两个视图的双目系统，且世界坐标系定在第一个视图： 

P=K[I|0]P′=K′[R|t]=K′R[I|−O′]则： P+=[K−10]O=(01)且： 

F=[P′C]×P′P+=[K′t]×K′RK−1=K′−T[t]×RK−1=K′−TR[RTt]xK−1=K′−TRKT[KRTt]×

对极点可以表示为： 

e=(−RTt1)=KRTte′=P′(01)=K′t

从而，F可以记作： F=[e′]×P′P+=K′−T[T]×RK−1=K′−TR[RTt]×K−1=K′−TRKT[e]×

**确定基础矩阵方法**：确定基础矩阵的最简单的方法即为8点法。由上可知，存在一对匹配点x1，x2，当有8对这样的点时如下图所示：

![img](https://github.com/zengqq1997/PCVch05-/blob/master/基础矩阵1.jpg)

则有如下方程：

![img](https://github.com/zengqq1997/PCVch05-/blob/master/基础矩阵2.jpg)

另左边矩阵为A，右边矩阵为f，即

![img](https://github.com/zengqq1997/PCVch05-/blob/master/基础矩阵3.jpg)

RANSAN算法可以用来消除错误匹配的的点，找到基础矩阵F，算法思想如下： 
（1）随机选择8个点； 
（2）用这8个点估计基础矩阵F； 
（3）用这个FF进行验算，计算用F验算成功的点对数n； 
重复多次，找到使n最大的F作为基础矩阵。

### 实例

基本步骤：首先是提取特征匹配，并估计基础矩阵和照相机矩阵。使用外极线作为第二个输入，通过在外极线上对每个特征点寻找最佳的匹配来找到更多的匹配。

**原图**

![img](https://github.com/zengqq1997/PCVch05-/blob/master/原图.jpg)

**特征提取部分代码**

```python
# coding: utf-8

# In[1]:
from pylab import *
from PIL import Image

# If you have PCV installed, these imports should work
from PCV.geometry import homography, camera, sfm
from PCV.localdescriptors import sift
from PIL import Image
from numpy import *
from pylab import *
import numpy as np
from PCV.tools import ransac


camera = reload(camera)
homography = reload(homography)
sfm = reload(sfm)
sift = reload(sift)

# Read features
im1 = array(Image.open('su01.jpg'))
sift.process_image('su01.jpg', 'im1.sift')

im2 = array(Image.open('su02.jpg'))
sift.process_image('su02.jpg', 'im2.sift')

l1, d1 = sift.read_features_from_file('im1.sift')
l2, d2 = sift.read_features_from_file('im2.sift')

matches = sift.match_twosided(d1, d2)
ndx = matches.nonzero()[0]

x1 = homography.make_homog(l1[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
x2 = homography.make_homog(l2[ndx2, :2].T)

d1n = d1[ndx]
d2n = d2[ndx2]
x1n = x1.copy()
x2n = x2.copy()

figure(figsize=(16,16))
sift.plot_matches(im1, im2, l1, l2, matches, True)
show()
```

**特征匹配结果图**

![img](https://github.com/zengqq1997/PCVch05-/blob/master/result1.jpg)

![img](https://github.com/zengqq1997/PCVch05-/blob/master/result11.jpg)

**估计基本矩阵和照相机矩阵代码**

```python
#def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-6):
def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-6):
    """ Robust estimation of a fundamental matrix F from point
    correspondences using RANSAC (ransac.py from
    http://www.scipy.org/Cookbook/RANSAC).

    input: x1, x2 (3*n arrays) points in hom. coordinates. """

    
    data = np.vstack((x1, x2))
    d = 10 # 20 is the original
    # compute F and return with inlier index
    F, ransac_data = ransac.ransac(data.T, model,
                                   8, maxiter, match_threshold, d, return_all=True)
    return F, ransac_data['inliers']

# find F through RANSAC
model = sfm.RansacModel()
F, inliers = F_from_ransac(x1n, x2n, model, maxiter=5000, match_threshold=1e-3)
print F

#计算照相机矩阵
P1 = array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = sfm.compute_P_from_fundamental(F)

print P2
```

> 这里使用的是RANSAC算法，从点对点中稳健地估计基础矩阵F



**估计基本矩阵结果**

![img](https://github.com/zengqq1997/PCVch05-/blob/master/Eresult.jpg)

![img](https://github.com/zengqq1997/PCVch05-/blob/master/Eresult1.jpg)

**和照相机矩阵结果**

![img](https://github.com/zengqq1997/PCVch05-/blob/master/Cresult.jpg)

![img](https://github.com/zengqq1997/PCVch05-/blob/master/Cresult1.jpg)

**三角剖分正确点，并计算每个照相机的深度代码**

```python
# triangulate inliers and remove points not in front of both cameras,三角剖分正确点，并计算每个照相机的深度
X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2)

# plot the projection of X绘制三维点
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2)
x1p = cam1.project(X)
x2p = cam2.project(X)

figure(figsize=(16, 16))
imj = sift.appendimages(im1, im2)
imj = vstack((imj, imj))

imshow(imj)

cols1 = im1.shape[1]
rows1 = im1.shape[0]
for i in range(len(x1p[0])):
    if (0<= x1p[0][i]<cols1) and (0<= x2p[0][i]<cols1) and (0<=x1p[1][i]<rows1) and (0<=x2p[1][i]<rows1):
        plot([x1p[0][i], x2p[0][i]+cols1],[x1p[1][i], x2p[1][i]],'c')
axis('off')
show()

d1p = d1n[inliers]
d2p = d2n[inliers]
```

**实验结果**

![img](https://github.com/zengqq1997/PCVch05-/blob/master/result2.jpg)

![img](https://github.com/zengqq1997/PCVch05-/blob/master/result22.jpg)

**外极线作为第二个输入代码**

```python
# Read features
im3 = array(Image.open('su03.jpg'))
sift.process_image('su03.jpg', 'im3.sift')
l3, d3 = sift.read_features_from_file('im3.sift')

matches13 = sift.match_twosided(d1p, d3)
ndx_13 = matches13.nonzero()[0]

x1_13 = homography.make_homog(x1p[:, ndx_13])
ndx2_13 = [int(matches13[i]) for i in ndx_13]
x3_13 = homography.make_homog(l3[ndx2_13, :2].T)


figure(figsize=(16, 16))
imj = sift.appendimages(im1, im3)
imj = vstack((imj, imj))

imshow(imj)

cols1 = im1.shape[1]
rows1 = im1.shape[0]
for i in range(len(x1_13[0])):
    if (0<= x1_13[0][i]<cols1) and (0<= x3_13[0][i]<cols1) and (0<=x1_13[1][i]<rows1) and (0<=x3_13[1][i]<rows1):
        plot([x1_13[0][i], x3_13[0][i]+cols1],[x1_13[1][i], x3_13[1][i]],'c')
axis('off')
show()

P3 = sfm.compute_P(x3_13, X[:, ndx_13])
```

**实验结果**

![img](https://github.com/zengqq1997/PCVch05-/blob/master/result3.jpg)

![img](https://github.com/zengqq1997/PCVch05-/blob/master/result33.jpg)

![img](https://github.com/zengqq1997/PCVch05-/blob/master/result4.jpg)

![img](https://github.com/zengqq1997/PCVch05-/blob/master/result44.jpg)

### 小结

**实验中遇到的问题**

![img](https://github.com/zengqq1997/PCVch05-/blob/master/error1.jpg)

在计算基本矩阵时，由于阈值设置得太小，导致没有符合的匹配点

**解决问题**：

![img](https://github.com/zengqq1997/PCVch05-/blob/master/solve.jpg)

将划红线的值改大，但是不能大于**1**
