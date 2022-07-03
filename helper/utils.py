import os

import scipy
import torch
import dgl
from dgl.data import RedditDataset
from dgl.distributed import partition_graph
import torch.distributed as dist
import time
from contextlib import contextmanager
from ogb.nodeproppred import DglNodePropPredDataset
import json
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_ogb_dataset(name):
    dataset = DglNodePropPredDataset(name=name, root='./dataset/')
    split_idx = dataset.get_idx_split()
    g, label = dataset[0]
    n_node = g.num_nodes()
    node_data = g.ndata
    node_data['label'] = label.view(-1).long()
    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][split_idx["train"]] = True
    node_data['val_mask'][split_idx["valid"]] = True
    node_data['test_mask'][split_idx["test"]] = True
    return g


def load_yelp():
    prefix = './dataset/yelp/'

    with open(prefix + 'class_map.json') as f:
        class_map = json.load(f)
    with open(prefix + 'role.json') as f:
        role = json.load(f)

    adj_full = scipy.sparse.load_npz(prefix + 'adj_full.npz')
    feats = np.load(prefix + 'feats.npy')
    n_node = feats.shape[0]

    g = dgl.from_scipy(adj_full)
    node_data = g.ndata

    label = list(class_map.values())
    node_data['label'] = torch.tensor(label)

    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][role['tr']] = True
    node_data['val_mask'][role['va']] = True
    node_data['test_mask'][role['te']] = True

    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['val_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['test_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['val_mask'], node_data['test_mask'])))
    assert torch.all(
        torch.logical_or(torch.logical_or(node_data['train_mask'], node_data['val_mask']), node_data['test_mask']))

    train_feats = feats[node_data['train_mask']]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

    node_data['feat'] = torch.tensor(feats, dtype=torch.float)

    return g


def load_data(dataset):
    if dataset == 'reddit':
        data = RedditDataset(raw_dir='./dataset/')
        g = data[0]
    elif dataset == 'ogbn-products':
        g = load_ogb_dataset('ogbn-products')
    elif dataset == 'ogbn-papers100m':
        g = load_ogb_dataset('ogbn-papers100M')
    elif dataset == 'yelp':
        g = load_yelp()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

    n_feat = g.ndata['feat'].shape[1]
    if g.ndata['label'].dim() == 1:
        n_class = g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]

    g.edata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, n_feat, n_class


def load_partition(args, rank):
    """
    Returns
    ----------
    DGLGraph
        the graph partition structure
    Dict[str, Tensor]
        including inner node features, labels, train_mask, etc.
    GraphPartitionBook
        The graph partition information, from the bigger picture (whole graph)
    """
    graph_dir = 'partitions/' + args.graph_name + '/'
    part_config = graph_dir + args.graph_name + '.json'

    print('loading partitions')

    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    node_type = node_type[0]
    node_feat[dgl.NID] = subg.ndata[dgl.NID]
    if 'part_id' in subg.ndata:
        node_feat['part_id'] = subg.ndata['part_id']
    node_feat['inner_node'] = subg.ndata['inner_node'].bool()
    node_feat['label'] = node_feat[node_type + '/label']
    node_feat['feat'] = node_feat[node_type + '/feat']
    node_feat['in_degree'] = node_feat[node_type + '/in_degree']
    node_feat['train_mask'] = node_feat[node_type + '/train_mask'].bool()
    node_feat.pop(node_type + '/label')
    node_feat.pop(node_type + '/feat')
    node_feat.pop(node_type + '/in_degree')
    node_feat.pop(node_type + '/train_mask')
    if not args.inductive:
        node_feat['val_mask'] = node_feat[node_type + '/val_mask'].bool()
        node_feat['test_mask'] = node_feat[node_type + '/test_mask'].bool()
        node_feat.pop(node_type + '/val_mask')
        node_feat.pop(node_type + '/test_mask')
    if args.dataset == 'ogbn-papers100m':
        node_feat.pop(node_type + '/year')
    subg.ndata.clear()
    subg.edata.clear()

    return subg, node_feat, gpb


def graph_partition(g, args):
    graph_dir = 'partitions/' + args.graph_name + '/'
    part_config = graph_dir + args.graph_name + '.json'

    # TODO: after being saved, a bool tensor becomes a uint8 tensor (including 'inner_node')
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_degree'] = g.in_degrees()
            partition_graph(g, args.graph_name, args.n_partitions, graph_dir, part_method=args.partition_method,
                            reshuffle=True, balance_edges=False, objtype=args.partition_obj)


def get_layer_size(n_feat, n_hidden, n_class, n_layers):
    layer_size = [n_feat]
    layer_size.extend([n_hidden] * (n_layers - 1))
    layer_size.append(n_class)
    return layer_size


def get_boundary(node_dict, gpb):
    """
    通过pytorch distributed point-to-point communication，使每个processor了解到其它的processor的boundary_nodes中，有哪些属于本processor。
    换句话说，让每个processor知道其它的任意processor需要它的哪些inner_nodes的信息。
    Parameters
    ----------
    node_dict
    gpb

    Returns
    -------
    boundary[i]表示rank i需要当前processor中的哪些inner_nodes节点来作为其boundary_nodes
    """
    rank, size = dist.get_rank(), dist.get_world_size()
    device = 'cuda'
    boundary = [None] * size

    for i in range(1, size):
        # left, right: rank左边（右边）第i个processor（processor之间是链式传播）
        left = (rank - i + size) % size
        right = (rank + i) % size
        belong_right = (node_dict['part_id'] == right)
        # num_right表示partition中有多少boundary_node属于右边第i个processor
        num_right = belong_right.sum().view(-1)
        if dist.get_backend() == 'gloo':
            num_right = num_right.cpu()
            num_left = torch.tensor([0])
        else:
            num_left = torch.tensor([0], device=device)
        # 每个processor向右边第i个processor发送其内部有多少个属于后者的节点，这些节点的信息之后需要从右侧的processor中拿
        # 每个processor接收从左边第i个processor的报告，得知后者内部有多少个属于前者的节点，这些节点需要当前processor发送给左侧
        req = dist.isend(num_right, dst=right)
        dist.recv(num_left, src=left)

        # partid2nids: given a partition id and return the inside node global ids
        # start: the start vertex in the i-th right processor, global id
        start = gpb.partid2nids(right)[0].item()
        # v: the local ids of boundary vertex which originally belong to the i-th right processor
        # v 是要发送到i-th right processor上去的
        v = node_dict[dgl.NID][belong_right] - start
        if dist.get_backend() == 'gloo':
            v = v.cpu()
            u = torch.zeros(num_left, dtype=torch.long)
        else:
            u = torch.zeros(num_left, dtype=torch.long, device=device)
        # 需要等上一次req发送完之后（关于有多少个boundary_nodes）再发送具体的boundary_nodes是那些
        req.wait()
        req = dist.isend(v, dst=right)

        dist.recv(u, src=left)
        # 因为每个processor内部存的boundary nodes并不是有序的，所以这里original processor收到数据后需要排个序
        u, _ = torch.sort(u)
        if dist.get_backend() == 'gloo':
            boundary[left] = u.cuda()
        else:
            boundary[left] = u
        req.wait()

    return boundary


def data_transfer(data, recv_shape, backend, dtype=torch.float, tag=0):
    """
    Parameters
    各processor将data（inner node的信息）发送给对应需要的processor
    data指明了当前processor需要发送给其它processor的信息；
    recv_shape指明了当前processor需要接受的来自其它processor的信息量
    ----------
    data：当前processor需要发给其它processor的信息
    recv_shape: 需要从其它每个processor获取的节点的数量
    backend
    dtype
    tag

    Returns
    -------

    """
    rank, size = dist.get_rank(), dist.get_world_size()
    res = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        if backend == 'gloo':
            # recv_shape[left]: 从左侧某processor中需要获取的节点的数量
            # data[left].shape[1]: 节点特征的维度。这里应该没有必要用left，因为所有节点的特征维度是一样的，用left只是顺便而已。
            res[left] = torch.zeros(torch.Size([recv_shape[left], data[left].shape[1]]), dtype=dtype)
        else:
            res[left] = torch.zeros(torch.Size([recv_shape[left], data[left].shape[1]]), dtype=dtype, device='cuda')

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        if backend == 'gloo':
            req = dist.isend(data[right].cpu(), dst=right, tag=tag)
        else:
            req = dist.isend(data[right], dst=right, tag=tag)
        # 注意recv是同步操作
        dist.recv(res[left], src=left, tag=tag)
        res[left] = res[left].cuda()
        req.wait()

    return res


def merge_feature(feat, recv):
    size = len(recv)
    for i in range(size - 1, 0, -1):
        if recv[i] is None:
            recv[i] = recv[i - 1]
            recv[i - 1] = None
    recv[0] = feat
    return torch.cat(recv)


def inductive_split(g):
    g_train = g.subgraph(g.ndata['train_mask'])
    g_val = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    g_test = g
    return g_train, g_val, g_test


def minus_one_tensor(size, device=None):
    if device is not None:
        return torch.zeros(size, dtype=torch.long, device=device) - 1
    else:
        return torch.zeros(size, dtype=torch.long) - 1


def nonzero_idx(x):
    """
    获取x中各非零项的第一维坐标，返回一个一行的tensor表示
    Parameters
    ----------
    x

    Returns
    -------

    """
    return torch.nonzero(x, as_tuple=True)[0]


def print_memory(s):
    torch.cuda.synchronize()
    print(s + ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024
    ))


@contextmanager
def timer(s):
    rank, size = dist.get_rank(), dist.get_world_size()
    t = time.time()
    yield
    print('(rank %d) running time of %s: %.3f seconds' % (rank, s, time.time() - t))
