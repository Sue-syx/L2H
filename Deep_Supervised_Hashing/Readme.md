##Deep Supervised Hashing  
  使用神经网络直接学习哈希编码，并用正则化方法将编码进行约束.    
* 让神经网络的输出通过正则的手法约束到 {-1，1} 之内.（后续使用 0 作为阈值进行离散化)
* 让网络的输出达到以下的要求: 相似的时候向量距离应该较近，反之则远 
* Siamese Network

###Loss Function  
**1.  单个Loss Func定义** ：   
![enter image description here](https://img-blog.csdn.net/20160908141822237)
**2. 总体Loss Func** ：  
![enter image description here](https://img-blog.csdn.net/20160908141831771)
**3. 优化** ：  
- 使用欧氏距离，使用额外的regularizer取代二值约束  
![enter image description here](https://img-blog.csdn.net/20160908141840503)
![enter image description here](https://img-blog.csdn.net/20160908141848503)
