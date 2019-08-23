## Deep Supervised Hashing  
* 使用神经网络直接学习哈希编码，并用正则化方法将编码进行约束.    
* 让神经网络的输出通过正则的手法约束到 {-1，1} 之内.（后续使用 0 作为阈值进行离散化)
* 让网络的输出达到以下的要求: 相似的时候向量距离应该较近，反之则远.
* Siamese Network

### Loss Function  
* 单个Loss Func定义 ：  
<div align=center><img width="429" height="129" src="https://img-blog.csdn.net/20160908141822237"/></div>  
* 总体Loss Func ：
<div align=center><img width="446" height="100" src="https://img-blog.csdn.net/20160908141831771"/></div>  
* 优化 ：  
  使用欧氏距离，使用额外的regularizer取代二值约束  
<div align=center><img width="460" height="129" src="https://img-blog.csdn.net/20160908141840503"/></div> <br>
<div align=center><img width="375" height="136" src="https://img-blog.csdn.net/20160908141848503"/></div>
