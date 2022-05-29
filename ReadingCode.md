一些数据结构

node_dict: 包含当然rank中全部节点（inner_node和boundary_node的信息）
* 以下的信息是针对partition内所有节点的(包括inner_node and boundary_node)
  * _ID: tensor, 包含所有的节点id(inner_node and boundary_node)
  * part_id: tensor，包含所有节点的所属partition id
  * inner_node: tensor (bool), 表示每一个节点是否是inner_node
* 以下信息仅针对partition的inner_node
  * label
  * feat
  * in_degree 
  * train_mask 