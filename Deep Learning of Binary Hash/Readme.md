## Deep Learning of Binary Hash Codes
[Deep Learning of Binary Hash Codes for Fast Image Retrieval　CVPR15](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W03/papers/Lin_Deep_Learning_of_2015_CVPR_paper.pdf)  
[> Code <](https://github.com/kevinlin311tw/caffe-cvprw15)
* Module1 - 在ImageNet上有监督地预训练CNN网络，以学习得到丰富的mid-level图像表示特征； 
* Module2 - 添加隐层(latent) ，通过在目标图像数据集finetuning网络，隐层可以学习到图像的 hashes-like 编码表示；
* Module3 - 利用 hashes-like 二值编码和 F7 层特征，采用 coarse-to-fine 策略检索相似图片.

<div align=center><img width=50% height=50% src="https://img-blog.csdn.net/20180518233641116?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L29KaU1vRGVZZTEyMzQ1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70"/></div> 

### Hash-like 二值编码学习
假设网络的最终输出分类层 F8 依赖于 h 个hidden attributes，各属性是 0 或 1(0表示不存在，1表示存在).  
如果图像的二值激活编码相似，其应该具有相同标签.
- 该隐层 H 是一个全连接层，其神经元激活情况由后面的 F8 层来控制，F8 层编码了图像语义并用于最终分类.
- 该隐层 H 不仅提供了 F7 层丰富特征的抽象表示，还联系着 mid-level 特征和 high-level 语义.
- 该隐层 H采用的是Sigmoid函数，以使激活值在 {0, 1} 之间.  
----
为了适应数据集，在目标数据集 fine-tune CNN网络.
- 初始权重设为ImageNet数据集预训练的CNN权重;
- 隐层 H 和最终分类层 F8 的权重采用随机初始化.  

### coarse-to-fine 检索方案  
- **粗糙检索** ：粗糙检索是用H层的二分哈希码，相似性用hamming距离衡量。待检索图像设为I，将I和所有的图像的对应H层编码进行比对后，选择出hamming距离小于一个阈值的m个构成一个池，其中包含了这m个比较相似的图像；  
- **细致检索** ：细致检索用到fc7层的特征，相似性用欧氏距离衡量。距离越小，则越相似。从粗糙检索得到的m个图像池中选出最相似的前k个图像作为最后的检索结果。
