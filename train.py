import torch.nn.functional as F
from module.model import *
from helper.utils import *
import torch.distributed as dist
import time
import copy
from multiprocessing.pool import ThreadPool
from sklearn.metrics import f1_score


def calc_acc(logits, labels):
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro')


@torch.no_grad()
def evaluate_induc(name, model, g, mode, result_file_name=None):
    """
    mode: 'val' or 'test'
    """
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    mask = g.ndata[mode + '_mask']
    logits = model(g, feat)
    logits = logits[mask]
    labels = labels[mask]
    acc = calc_acc(logits, labels)
    buf = "{:s} | Accuracy {:.2%}".format(name, acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, acc


@torch.no_grad()
def evaluate_trans(name, model, g, result_file_name=None):
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    val_mask, test_mask = g.ndata['val_mask'], g.ndata['test_mask']
    logits = model(g, feat)
    val_logits, test_logits = logits[val_mask], logits[test_mask]
    val_labels, test_labels = labels[val_mask], labels[test_mask]
    val_acc = calc_acc(val_logits, val_labels)
    test_acc = calc_acc(test_logits, test_labels)
    buf = "{:s} | Validation Accuracy {:.2%} | Test Accuracy {:.2%}".format(name, val_acc, test_acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, val_acc


def average_gradients(model, n_train):
    reduce_time = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        t0 = time.time()
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= n_train
        reduce_time += time.time() - t0
    return reduce_time


def move_to_cuda(graph, part, node_dict):

    for key in node_dict.keys():
        node_dict[key] = node_dict[key].cuda()
    graph = graph.int().to(torch.device('cuda'))
    part = part.int().to(torch.device('cuda'))

    return graph, part, node_dict


def get_pos(node_dict, gpb):
    """
    对于world_size==4的情况，生成4个tensor，其中pos[rank]=None, rank为当前processor（称作current rank）。
    返回的每个tensor记录了一个boundary node从它的root rank的local id到current rank的映射。
    例如，当current rank == 2时，pos[0][23796] == tensor(37289,)表示，rank 0中local id为23796的节点是rank 0的inner node，是rank 2的boundary node，这个节点在rank 2中的local id为37289.
    这个函数的作用是，了解其它processor的inner node在当前processor中存储的index是多少
    Parameters
    ----------
    node_dict
    gpb

    Returns
    -------
    """
    pos = []
    rank, size = dist.get_rank(), dist.get_world_size()
    for i in range(size):
        if i == rank:
            pos.append(None)
        else:
            # rank i中inner nodes的数量
            part_size = gpb.partid2nids(i).shape[0]
            # rank i的起始点的id
            start = gpb.partid2nids(i)[0].item()
            p = minus_one_tensor(part_size, 'cuda')
            # 生成一个tensor，包含当前rank (processor) 中的boundary nodes中属于rank i的那些节点的序号（current rank local id）
            in_idx = nonzero_idx(node_dict['part_id'] == i)
            # 将上面生成的tensor的节点转化成global id之后再减去rank i的start得到该节点在rank i中的local id
            out_idx = node_dict[dgl.NID][in_idx] - start
            p[out_idx] = in_idx
            pos.append(p)
    return pos


def get_recv_shape(node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    recv_shape = []
    for i in range(size):
        if i == rank:
            recv_shape.append(None)
        else:
            t = (node_dict['part_id'] == i).int().sum().item()
            recv_shape.append(t)
    return recv_shape


def create_inner_graph(graph, node_dict):
    """
    Extract (u, v) such that both u and v are inner nodes of partitioned subgraph
    """
    u, v = graph.edges()
    sel = torch.logical_and(node_dict['inner_node'].bool()[u], node_dict['inner_node'].bool()[v])
    u, v = u[sel], v[sel]
    return dgl.graph((u, v))


def order_graph(part, graph, gpb, node_dict, pos):
    rank, size = dist.get_rank(), dist.get_world_size()
    one_hops = []
    for i in range(size):
        if i == rank:
            one_hops.append(None)
            continue
        start = gpb.partid2nids(i)[0].item()
        nodes = node_dict[dgl.NID][node_dict['part_id'] == i] - start
        nodes, _ = torch.sort(nodes)
        one_hops.append(nodes)
    # one_hops: one_hops[i]包括了当前processor的boundary_nodes中，属于rank i的那些节点在rank i中的local id
    return construct(part, graph, pos, one_hops)


def move_train_first(graph, node_dict, boundary):
    """
    按照是否是训练节点对inner nodes排序，训练节点在前，非训练节点在后。并同时根据new id更改对应的node_dict和boundary的信息。
    这里没有move cuda相关的操作。
    Parameters
    ----------
    graph
    node_dict
    boundary

    Returns
    -------

    """
    train_mask = node_dict['train_mask']
    # 当前processor中需要训练的节点数量
    num_train = torch.count_nonzero(train_mask).item()
    # 只统计V类型的点：V类型的点是inner_nodes，U类型的点有inner nodes也有boundary nodes
    # 也有可能某个inner node是孤立节点，在之前的partition graph里是有这种节点的信息，但在现在的order graph中，这部分节点不保存
    # 这里的num_tot就是当前processor中非孤立inner nodes的数量
    num_tot = graph.num_nodes('_V')

    # 对所有inner nodes重排序，分配新的id。训练节点放在前，非训练节点放在后。
    new_id = torch.zeros(num_tot, dtype=torch.int, device='cuda')
    # 为每一个训练节点分配一个新id
    new_id[train_mask] = torch.arange(num_train, dtype=torch.int, device='cuda')
    # 为inner graph中非训练节点分配后续的id
    new_id[torch.logical_not(train_mask)] = torch.arange(num_train, num_tot, dtype=torch.int, device='cuda')

    # 转换成新的id表示。
    # TODO: 转换方式没有看懂
    u, v = graph.edges()
    u[u < num_tot] = new_id[u[u < num_tot].long()]
    v = new_id[v.long()]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})

    for key in node_dict:
        node_dict[key][new_id.long()] = node_dict[key][0:num_tot].clone()

    for i in range(len(boundary)):
        if boundary[i] is not None:
            boundary[i] = new_id[boundary[i]].long()

    return graph, node_dict, boundary


def create_graph_train(graph, node_dict):
    u, v = graph.edges()
    num_u = graph.num_nodes('_U')
    sel = nonzero_idx(node_dict['train_mask'][v.long()])
    u, v = u[sel], v[sel]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if graph.num_nodes('_U') < num_u:
        graph.add_nodes(num_u - graph.num_nodes('_U'), ntype='_U')
    return graph, node_dict['in_degree'][node_dict['train_mask']]


def precompute(graph, node_dict, boundary, recv_shape, args):
    """
    pre-process the feature of partition inner nodes
    对所有V进行一次mean aggregation。
    因为GraphSAGE在训练过程中，第0层首先开始一次聚合，这个可以在每次训练过程中算，但也可以在训练之前先对所有的节点进行一次预计算，算出第0层这个mean aggregation的聚合结果，之后再使用NN变换以及后面各层的训练。
    """
    rank, size = dist.get_rank(), dist.get_world_size()
    in_size = node_dict['inner_node'].bool().sum()
    feat = node_dict['feat']
    send_info = []
    for i, b in enumerate(boundary):
        if i == rank:
            send_info.append(None)
        else:
            send_info.append(feat[b])
    # 交换inner node信息。当前processor发送给其它processor，并从其它processor处接受boundary nodes的特征信息，即为返回的recv_feat。
    recv_feat = data_transfer(send_info, recv_shape, args.backend, dtype=torch.float)
    if args.model == 'graphsage':
        with graph.local_scope():
            graph.nodes['_U'].data['h'] = merge_feature(feat, recv_feat)
            graph['_E'].update_all(fn.copy_src(src='h', out='m'),
                                   fn.sum(msg='m', out='h'),
                                   etype='_E')
            mean_feat = graph.nodes['_V'].data['h'] / node_dict['in_degree'][0:in_size].unsqueeze(1)
        return torch.cat([feat, mean_feat[0:in_size]], dim=1)
    else:
        raise Exception


def create_model(layer_size, args):
    if args.model == 'graphsage':
        return GraphSAGE(layer_size, F.relu, args.use_pp, norm=args.norm, dropout=args.dropout,
                         n_linear=args.n_linear, train_size=args.n_train)
    else:
        raise NotImplementedError


def reduce_hook(param, name, n_train):
    def fn(grad):
        ctx.reducer.reduce(param, name, grad, n_train)
    return fn


def construct(part, graph, pos, one_hops):
    """
    把整个partition内的点、边进行重新排序。边的顺序是首先inner graph的边，然后是和processor 0相关的节点、边。节点的local id顺序也是按照这种优先级进行的重排序。
    Parameters
    ----------
    part: inner_nodes生成的子图，inner graph
    graph: partition内所有节点生成的子图
    pos: 其它rank的inner_nodes which are boundary nodes in current rank，指示其它processor中的inner_node在当前processor中的index是多少（是当前processor的boundary node）
    one_hops: one_hops[i]包括了当前processor的boundary_nodes中，属于rank i的那些节点在rank i中的local id。和pos是对应的关系

    Returns
    -------

    """
    rank, size = dist.get_rank(), dist.get_world_size()
    # inner-graph节点数量
    tot = part.num_nodes()
    u, v = part.edges()
    # 先添加inner-graph的边
    u_list, v_list = [u], [v]
    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            # 下面的等式左边的u，就是另一个processor的inner node分别在当前processor中的index。是按照inner node的顺序排列的。
            u = pos[i][u]
            # 形成一个列表，对应所有的boundary_nodes形成的src节点
            u_ = torch.repeat_interleave(graph.out_degrees(u.int()).long()) + tot
            # tot表示为中心节点的数量和与processor i相关的boudary nodes的数量之和
            tot += u.shape[0]
            # 这里得到的v应该都是inner nodes
            _, v = graph.out_edges(u.int())
            u_list.append(u_.int())
            v_list.append(v)
    u = torch.cat(u_list)
    v = torch.cat(v_list)
    # 异质图，src nodes的类型是 _U，dest nodes类型是_V，边的类型是_E。这部分是dgl的定义
    # 这里的v类型的节点应该都是inner node
    g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if g.num_nodes('_U') < tot:
        # 看起来正常的情况不会进入这个分支
        g.add_nodes(tot - g.num_nodes('_U'), ntype='_U')
    return g


def extract(graph, node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    sel = (node_dict['part_id'] < size)
    for key in node_dict.keys():
        if node_dict[key].shape[0] == sel.shape[0]:
            node_dict[key] = node_dict[key][sel]
    graph = dgl.node_subgraph(graph, sel, store_ids=False)
    return graph, node_dict


def run(graph, node_dict, gpb, args):
    """
    Parameters:
    ---
    graph: DGLGraph
        The subgraph belongs to current partition
    node_dict: Dict[str, Tensor]
        The node feature and all other information in the corresponding subgraph

    """
    
    rank, size = dist.get_rank(), dist.get_world_size()

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    if rank == 0 and args.eval:
        full_g, n_feat, n_class = load_data(args.dataset)
        if args.inductive:
            _, val_g, test_g = inductive_split(full_g)
        else:
            val_g, test_g = full_g.clone(), full_g.clone()
        del full_g

    if rank == 0:
        os.makedirs('checkpoint/', exist_ok=True)
        os.makedirs('results/', exist_ok=True)

    part = create_inner_graph(graph.clone(), node_dict)
    num_in = node_dict['inner_node'].bool().sum().item()
    part.ndata.clear()
    part.edata.clear()

    print(f'Process {rank} has {graph.num_nodes()} nodes, {graph.num_edges()} edges '
          f'{part.num_nodes()} inner nodes, and {part.num_edges()} inner edges.')

    graph, part, node_dict = move_to_cuda(graph, part, node_dict)
    # 让当前processor了解其inner_nodes的信息在其它processor中的依赖情况，即哪些inner_nodes信息需要发给哪个processor。
    boundary = get_boundary(node_dict, gpb)

    layer_size = get_layer_size(args.n_feat, args.n_hidden, args.n_class, args.n_layers)

    # 其它processor中inner_node local id到当前processor中boundary_node local id的映射
    pos = get_pos(node_dict, gpb)
    # order graph是图结构
    graph = order_graph(part, graph, gpb, node_dict, pos)
    in_deg = node_dict['in_degree']

    # graph是图结构，node_dict是节点的特征，boundary标记了训练过程中数据的传输
    # 将partition内的inner node进行一次重排序，并更改对应的信息
    graph, node_dict, boundary = move_train_first(graph, node_dict, boundary)

    # 获取需要从其它的processor中获取的节点信息的数量。即当前processor中有多少个boundary node属于其它任一processor i
    recv_shape = get_recv_shape(node_dict)

    ctx.buffer.init_buffer(num_in, graph.num_nodes('_U'), boundary, recv_shape, layer_size[:args.n_layers-args.n_linear],
                           use_pp=args.use_pp, backend=args.backend, pipeline=args.enable_pipeline,
                           corr_feat=args.feat_corr, corr_grad=args.grad_corr, corr_momentum=args.corr_momentum)

    if args.use_pp:
        node_dict['feat'] = precompute(graph, node_dict, boundary, recv_shape, args)

    labels = node_dict['label'][node_dict['train_mask']]
    train_mask = node_dict['train_mask']
    part_train = train_mask.int().sum().item()

    del boundary
    del part
    del pos

    torch.manual_seed(args.seed)
    model = create_model(layer_size, args)
    model.cuda()

    ctx.reducer.init(model)

    for i, (name, param) in enumerate(model.named_parameters()):
        param.register_hook(reduce_hook(param, name, args.n_train))

    best_model, best_acc = None, 0

    if args.grad_corr and args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_grad_feat.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    elif args.grad_corr:
        result_file_name = 'results/%s_n%d_p%d_grad.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    elif args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_feat.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    else:
        result_file_name = 'results/%s_n%d_p%d.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    if args.dataset == 'yelp':
        loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_dur, comm_dur, reduce_dur = [], [], []
    torch.cuda.reset_peak_memory_stats()
    thread = None
    pool = ThreadPool(processes=1)

    feat = node_dict['feat']

    node_dict.pop('train_mask')
    node_dict.pop('inner_node')
    node_dict.pop('part_id')
    node_dict.pop(dgl.NID)

    if not args.eval:
        node_dict.pop('val_mask')
        node_dict.pop('test_mask')

    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        if args.model == 'graphsage':
            logits = model(graph, feat, in_deg)
        else:
            raise Exception
        if args.inductive:
            loss = loss_fcn(logits, labels)
        else:
            loss = loss_fcn(logits[train_mask], labels)
        del logits
        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        ctx.buffer.next_epoch()

        pre_reduce = time.time()
        ctx.reducer.synchronize()
        reduce_time = time.time() - pre_reduce
        optimizer.step()

        if epoch >= 5 and epoch % args.log_every != 0:
            train_dur.append(time.time() - t0)
            comm_dur.append(ctx.comm_timer.tot_time())
            reduce_dur.append(reduce_time)

        if (epoch + 1) % 10 == 0:
            print("Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Loss {:.4f}".format(
                  rank, epoch, np.mean(train_dur), np.mean(comm_dur), np.mean(reduce_dur), loss.item() / part_train))

        ctx.comm_timer.clear()

        del loss

        if rank == 0 and args.eval and (epoch + 1) % args.log_every == 0:
            if thread is not None:
                model_copy, val_acc = thread.get()
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = model_copy
            model_copy = copy.deepcopy(model)
            if not args.inductive:
                thread = pool.apply_async(evaluate_trans, args=('Epoch %05d' % epoch, model_copy,
                                                                val_g, result_file_name))
            else:
                thread = pool.apply_async(evaluate_induc, args=('Epoch %05d' % epoch, model_copy,
                                                                val_g, 'val', result_file_name))

    if args.eval and rank == 0:
        if thread is not None:
            model_copy, val_acc = thread.get()
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model_copy
        torch.save(best_model.state_dict(), 'model/' + args.graph_name + '_final.pth.tar')
        print('model saved')
        print("Validation accuracy {:.2%}".format(best_acc))
        _, acc = evaluate_induc('Test Result', best_model, test_g, 'test')


def check_parser(args):
    if args.norm == 'none':
        args.norm = None


def init_processes(rank, size, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '%d' % args.port
    dist.init_process_group(args.backend, rank=rank, world_size=size)
    rank, size = dist.get_rank(), dist.get_world_size()
    check_parser(args)
    g, node_dict, gpb = load_partition(args, rank)
    run(g, node_dict, gpb, args)
