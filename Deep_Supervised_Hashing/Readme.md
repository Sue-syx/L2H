## Deep Supervised Hashing 
[Deep Supervised Hashing for Fast Image Retrieval　CVPR16](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf)
* 使用神经网络直接学习哈希编码，并用正则化方法将编码进行约束.    
* 让神经网络的输出通过正则的手法约束到 {-1，1} 之内.（后续使用 0 作为阈值进行离散化)
* 让网络的输出达到以下的要求: 相似的时候向量距离应该较近，反之则远.
* Siamese Network  

<div align=center><img width=60% height=60% src="https://img-blog.csdn.net/20160908141235901"/></div> 

### Loss Function
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
