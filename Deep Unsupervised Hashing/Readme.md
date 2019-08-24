## Deep Unsupervised Hashing 
[Learning Compact Binary Descriptors with Unsupervised Deep Neural Networks　CVPR16](https://www.iis.sinica.edu.tw/~kevinlin311.tw/cvpr16-deepbit.pdf)  
  
利用从ImageNet上预训练的中间层的图像表示( mid-level image representation )  
无监督学习到了二值描述子(DeepBit), 主要设计三个内容 ：  
* minimal loss quantization 
* evenly distributed codes  
* uncorrelated bits  
<div align=center><img width=60% height=60% src="https://img-blog.csdn.net/20180506192556259?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNDE3Mjg3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70"/></div>  

> #### Unsupervised Hashing:  
> - Locality Sensitive Hashing (LSH)：使用随机投影的方式将原始数据映射到一个低维度的特征空间，然后对其二值化  
> - Semantic hashing (SH)：建立多层的Restricted Boltzmann Machines (RBM) 学习得到进制二进制码 (针对文本)
> - Spectral hashing (SpeH): 谱分割方法生成二值码  
> - Iterative qauntization (ITQ)：使用迭代优化策略找到二值损失最小时的投影 

### Overall Learning Objectives  
<div align=center><img width=35% height=35% src="https://img-blog.csdn.net/20180506194008208?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNDE3Mjg3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70 "/></div>  

### 1. Learning Discriminative Binary Descriptors
deepbit的目标是找到投影函数可以将输入图像映射到一个二值数据中，同时保留原始图像的具有区分性的信息。  
量化损失越小，二值描述子保留原始图像信息的效果越好，也就是越接近原始投影值  
<div align=center><img width=35% height=35% src="https://img-blog.csdn.net/20180506195126790?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNDE3Mjg3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70"/></div>  
  
### 2. Learning Efficient Binary Descriptors  
尽可能的使二值码均匀分布，熵越大，能够表达的信息越多 ，以 50% 分界  
<div align=center><img width=30% height=30% src="https://img-blog.csdn.net/20180506195432900?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNDE3Mjg3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70"/></div>  
其中 <div align=center><img width=35% height=35% src="https://img-blog.csdn.net/20180506195626443?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNDE3Mjg3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70"/></div>  

### 3. Learning Rotation Invariant Binary Descriptors  
我们希望得到的描述能具有旋转不变性， 
estimation error 可能会随着角度增大而变得很大，所以增加了一个惩罚项C(θ)
<div align=center><img width=30% height=30% src="https://img-blog.csdn.net/20180506200031963?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNDE3Mjg3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70"/></div>  
所以最小化函数： <div align=center><img width=35% height=35% src="https://img-blog.csdn.net/20180506195708745?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNDE3Mjg3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70"/></div>
