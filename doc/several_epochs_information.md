## PipeGCN及Fine-grained pipeline设计
p.s. 该方法不是真正的fine-grained，因为没有设置SAGA四个阶段。SAGA的操作需要在DGL中设置。
### PipeGCN

代码层面，训练过程中一层中的操作：

```
for each layer:
	update_feature()
	SAGELayer()
```

* update_feature(): 从feature buffer中获取上一个epoch中其它processor传过来的boundary nodes feature，并和当前processor中inner node在当前epoch的feature做拼接，最后返回拼接形成的用于下一层训练输入的feature
* SAGELayer(): 下一层的关于GraphSAGE的部分，包括aggregation和linear transformation操作

关于update feature函数：

```
update_feature():
	Sync()
	feat_concat()
	async(&feat_transfer)
	register_hook(&grad_hook)
	return
```

* Sync(): 同步点，确保当前层的buffer传输已结束，即已接收完全上一个epoch的信息
* feat_concat()：将buffer中的boundary nodes stale feature和inner node feature组合在一起，形成训练中实际用到的feature
* async: 异步调用feat_transfer()函数
* register_hook: 定义反向传播时的操作
* feat_transfer: 将当前epoch的inner node feature根据之前统计的boundary发送给相应的processor，用于其下一个epoch的训练
* grad_hook: 封装了反向传播时需要做的梯度更新操作。对应前向计算中的update_feature()函数
  * grad_hook的逻辑与update_feature()相似，仅传播方向相反
  * grad_hook中同样涉及到grad_transfer()操作，与前向计算中feat_transfer()相对应

### PipeGCN与Fine-grained Pipeline (以Dorylus为例) 的区别

* 编程模型（流水线阶段）的差异：Fine-grained使用SAGA-NN模型。SAGA能更好抽象GAT这样的GNN模型，但我们当前先关注GCN, GraphSAGE的情况下可以暂时不考虑在DGL中实现SAGA
* 流水线结构：在每一层FW(forward), BW(backward)阶段，Fine-grained都可以根据邻居节点的传输状态决定使用哪个epoch的feature
* 系统结构
  * PipeGCN这样的GPU cluster是去中心化的训练和参数更新
  * Dorylus是中心化的
    * Dorylus有Graph Server (GS)，Parameter Server（PS），每个computation node不需要维护大量的信息（特征、图结构、参数等）
    * Graph Server做aggregation的操作。GS在进行该操作时会实时从其它GS获取邻居节点信息。PipeGCN是基于feature_buffer和提前传输来做本地缓存。
* 额外一点：PipeGCN是full-graph training, Dorylus不是。这样使得PipeGCN通过feature_buffer来缓存比较有效，因为所有boundary_nodes都会用到。

### 修改PipeGCN的结构使其支持Fine-grained Pipeline

综上，从训练模式（而非硬件系统结构）的角度来看，主要的两点不同：

* SAGA-NN模型
* 可以使用更多epoch之前的staleness function，且是灵活的结构，而非固定使用n个epoch前的数据

从实现的角度，这里先不考虑SAGA的相关问题，仅考虑实现第二点，即灵活使用多个epoch之前的数据。

修改后的GPU存储结构示意图：

![image-20220608142241024](https://s2.loli.net/2022/06/08/HgjRxyz1DKdwEqJ.png)

说明：

当前的PipeGCN的一个processor的feature buffer是为每一层预留buffer，分别为FW存储feature和BW存储gradient。在去中心化的模式下需要使用多个epoch之前的数据，直接的方式是在GPU中预留多个epoch的buffer。同时，为每层，FW/BW各设置一个latest指针，指向当前已全部传输完成的最近的epoch的boudary node features。即用latest指针指向的epoch中的feature来作为stale feature。

算法流程：

```
update_feature():
	check_latest_in_s()
	staleFeat = &latest
	feat_concat(staleFeat)
	clear_buffer()
	async(&feat_transfer)
	register_hook(&grad_hook)
	return
```

P.S. 为每一个processor设置一个守护进程，检查最近的接收完全的epoch buffer，并将latest指针指向这个buffer

* check_latest_in_s()：检查latest指针指向的epoch是否在bounded staleness允许范围之内；若不在，则等待
* clear_buffer()：将比latest指向的feature buffer更旧的feature buffer清空
* grad_hook中也需要做与update_feature相同的改变，以利用多个epoch之前的gradients

这种方法存在的一些问题：

* 与Dorylus相比
  * full-graph training方式通信开销更大，即全部boundary nodes信息都需要传输。
* 与PipeGCN相比
  * GPU内存开销增大

该方法的优势：

* 与Dorylus相比
  * full-graph收敛准确率高
  * GPU cluster模式也许比Serverless方式性能好
* 与PipeGCN相比
  * 更灵活的流水线，允许使用多个epoch之前的信息。或许可以减少阻塞：PipeGCN固定使用一个epoch之前的信息，当需要用feature而前一个epoch还未传输完时会造成阻塞（update_feature()中的Sync()阶段）。而新的方法可以使用更久之前的feature，只要latest在S个epoch之内即可。
