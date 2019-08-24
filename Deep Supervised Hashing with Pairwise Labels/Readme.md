## Deep Supervised Hashing with Pairwise Labels
[Feature Learning Based Deep Supervised Hashing with Pairwise Label　IJCAI16](https://www.ijcai.org/Proceedings/16/Papers/245.pdf)  
  
DPSH(deep pairwise-supervised hashing)，通过深度学习从pairwise label中学习图像的特征和hash code.  
使用CNN-F网络模型，上下两个CNN使用相同的网络结构，共享同样的权重 (Siamese Network).
* 通过CNN的conv层学习图像特征
* 使用网络的全连接层学习hash function
* 设计合理的loss function使pairwise中label相似时，hash code尽可能相似。即相似的label，hash code之间的海明距离小；不相似的label，hash code 之间海明距离大。
<div align=center><img width=40% height=40% src="https://img-blog.csdn.net/20161018152020051"/></div> 

### Object Function  
给定所有点的二进制码B=\{ b_{i} \}_{i=1}^n, 点对之间的似然函数可以定义如下：　　
$ps_{ij}|B=\{^{\sigma(\Omega_{ij}),　　　　 s_{ij}=1} _{1-\sigma(\Omega_{ij}),\qquad s_{ij}=0}$
<ul>
    <li>单个Loss Func定义 ： </li>
      <div align=center><img width=40% height=40% src="https://img-blog.csdn.net/20160908141822237"/></div> 
      <br>
    <li>总体Loss Func ：</li>
      <div align=center><img width=40% height=40% src="https://img-blog.csdn.net/20160908141831771"/></div>  
      <br>
    <li>优化(European distance + regularizer) ：</li>
      <div align=center><img width=40% height=40% src="https://img-blog.csdn.net/20160908141840503"/></div> <br>
      <div align=center><img width=40% height=40% src="https://img-blog.csdn.net/20160908141848503"/></div>
</ul>
