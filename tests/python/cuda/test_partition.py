import torch
from quiver.utils import cal_memory_budget_to_bytes
import shutil
import os

import quiver
from ogb.lsc import MAG240MDataset
import os.path as osp
import random
from ogb.nodeproppred import PygNodePropPredDataset
from matplotlib import pyplot as plt
from torch_geometric.datasets import Reddit


QUIVER_MAGIC_NUMBER = 64

def partition_without_replication(probs):
    """Partition node with node access distribution. 
    The result will cause no replication between each parititon.

    Args:
        probs (torch.Tensor): node access distribution

    Returns:
        [torch.Tensor]: list of IDs for each partition

    """

    device = torch.cuda.current_device()
    partitioned_num = len(probs)

    probs = [prob.to(device) for prob in probs]
    total_node_num = probs[0].size(0)

    res = [[] for _ in range(partitioned_num)]

    chunk_size = QUIVER_MAGIC_NUMBER * partitioned_num
    chunk_num = (total_node_num + chunk_size - 1) // chunk_size

    current_chunk_start_pos = 0
    current_partition_idx = 0
    for _ in range(chunk_num):
        current_chunk_end_pos = min(total_node_num, current_chunk_start_pos + chunk_size)
        current_chunk_size = current_chunk_end_pos - current_chunk_start_pos
        chunk = torch.arange(current_chunk_start_pos, current_chunk_end_pos, device=device)
        probs_sum_chunk = [
            torch.zeros(current_chunk_size, device=device) + 1e-6 for _ in range(partitioned_num)
        ]
        for src_rank in range(partitioned_num):
            for dst_rank in range(partitioned_num):
                if dst_rank == src_rank:
                    probs_sum_chunk[src_rank] += probs[dst_rank][chunk] * partitioned_num
                else:
                    probs_sum_chunk[src_rank] -= probs[dst_rank][chunk]
        assigned_node_size = 0
        per_partition_size = QUIVER_MAGIC_NUMBER
        for partition_idx in range(current_partition_idx, current_partition_idx + partitioned_num):
            partition_idx = partition_idx % partitioned_num
            actual_per_partition_size = min(per_partition_size, current_chunk_size - assigned_node_size)
            _, sorted_res_order = torch.sort(probs_sum_chunk[partition_idx], descending=True)
            pick_chunk_part = sorted_res_order[:actual_per_partition_size]
            pick_ids = chunk[pick_chunk_part]
            res[partition_idx].append(pick_ids)
            probs_sum_chunk[partition_idx][pick_chunk_part] = -1
            assigned_node_size += actual_per_partition_size
        current_partition_idx += 1
        current_chunk_start_pos += current_chunk_size

    for partition_idx in range(partitioned_num):
        res[partition_idx] = torch.cat(res[partition_idx])
    return res, probs


def quiver_partition(probs:torch.Tensor, result_path: str, cache_memory_budget=0, per_feature_size=0):
    """
    Partition graph topology based on access probability

    Args:
        probs:
        result_path (str): path for partition result
        cache_memory_budget (Union[str, int, float]): user-specified memory budget for caching hot feature
        per_feature_size (Union[str, int, float]): per-feature size for user's feature
    
    Returns:
        partition_res (torch.Tensor): partitioned result
        cache_res (torch.Tensor): cached result
    """

    if os.path.exists(result_path):
        res = input(f"{result_path} already exists, enter Y/N to continue, If continue, {result_path} will be deleted:")
        res = res.upper()
        if res == "Y":
            shutil.rmtree(result_path)
        else:
            print("exiting ...")
            exit()
    
    partition_num = len(probs)
        
    
    # create result folder
    for partition_idx in range(partition_num):
        os.makedirs(os.path.join(result_path, f"partition_{partition_idx}"))
    
    # calculate cached feature count
    cache_memory_budget_bytes = cal_memory_budget_to_bytes(cache_memory_budget)
    per_feature_size_bytes = cal_memory_budget_to_bytes(per_feature_size)
    cache_count = int(cache_memory_budget_bytes / (per_feature_size_bytes + 1e-6))
    per_partition_cache_count = cache_count // partition_num

    partition_res, changed_probs = partition_without_replication(probs)
    cache_res = [None] * partition_num

    if cache_count > 0:
        for partition_idx in range(partition_num):
            _, prev_order = torch.sort(changed_probs[partition_idx], descending=True)
            cache_res[partition_idx] = prev_order[: per_partition_cache_count]
    
    for partition_idx in range(partition_num):
        partition_result_path = os.path.join(result_path, f"partition_{partition_idx}", "partition_res.pth")
        cache_result_path = os.path.join(result_path, f"partition_{partition_idx}", "cache_res.pth")
        torch.save(partition_res[partition_idx], partition_result_path)
        torch.save(cache_res[partition_idx], cache_result_path)

    return partition_res, cache_res


def load_partition(partition_idx: int, result_path:str):
    """
    Load partition result for partition ${partition_idx}

    Args:
        partition_idx (int): Partition idx
        partition_result_path (str): partition result path
    """

    if not os.path.exists(result_path):
        raise Exception("Result path not exists")
    
    partition_result_path = os.path.join(result_path, f"partition_{partition_idx}", "partition_res.pth")
    cache_result_path = os.path.join(result_path, f"partition_{partition_idx}", "cache_res.pth")

    partition_res = torch.load(partition_result_path)
    cache_res = torch.load(cache_result_path)

    return partition_res, cache_res

def load_mag240M():
    indptr = torch.load("/data/mag/mag240m_kddcup2021/csr/indptr.pt")
    indices = torch.load("/data/mag/mag240m_kddcup2021/csr/indices.pt")
    dataset = MAG240MDataset("/data/mag")
    train_idx = torch.from_numpy(dataset.get_idx_split('train'))
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")
    print(f"average degree of MAG240M = {torch.sum(csr_topo.degree) / csr_topo.node_count}")
    return train_idx, csr_topo, quiver_sampler

def load_paper100M():
    indptr = torch.load("/data/papers/ogbn_papers100M/csr/indptr_bi.pt")
    indices = torch.load("/data/papers/ogbn_papers100M/csr/indices_bi.pt")
    train_idx = torch.load("/data/papers/ogbn_papers100M/index/train_idx.pt")
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")
    print(f"average degree of paper100M = {torch.sum(csr_topo.degree) / csr_topo.node_count}")
    return train_idx, csr_topo, quiver_sampler

def load_products():
    root = "/home/dalong/data/products/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")
    print(f"average degree of products = {torch.sum(csr_topo.degree) / csr_topo.node_count}")
    return train_idx, csr_topo, quiver_sampler

def load_reddit():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 10], 0, mode="UVA")
    print(f"average degree of Reddit = {torch.sum(csr_topo.degree) / csr_topo.node_count}")
    return train_idx, csr_topo, quiver_sampler

def load_com_lj():
    indptr = torch.load("/home/dalong/data/com-lj_indptr.pt")
    indices = torch.load("/home/dalong/data/com-lj_indices.pt")
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    print("node count", csr_topo.node_count)
    print("edge count", csr_topo.edge_count)
    
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")
    train_idx = torch.randint(0, csr_topo.node_count, (csr_topo.node_count // 5, ))
    print(f"average degree of LJ = {torch.sum(csr_topo.degree) / csr_topo.node_count}")
    return train_idx, csr_topo, quiver_sampler

def test_cdf():


    #train_idx, csr_topo, quiver_sampler = load_mag240M()
    #train_idx, csr_topo, quiver_sampler = load_paper100M()
    train_idx, csr_topo, quiver_sampler = load_com_lj()
    #train_idx, csr_topo, quiver_sampler = load_reddit()

    for partition_num in range(1, 10, 1):
        idx_len = train_idx.size(0)
        hit_cum = torch.zeros_like(csr_topo.indptr)
        random.shuffle(train_idx)
        real_train_idx = train_idx[:idx_len // partition_num]

        train_loader = torch.utils.data.DataLoader(real_train_idx,
                                                batch_size=1024,
                                                pin_memory=True,
                                                shuffle=True)
        for epoch in range(5):
            for seeds in train_loader:
                n_id, _, _ = quiver_sampler.sample(seeds)
                hit_cum[n_id] += 1
            #print(f"Done {epoch}/{partition_num}")

        
        sorted_hit_cum, _ = torch.sort(hit_cum, descending=True)
        total_hit = torch.sum(sorted_hit_cum)
        levels = []
        x = []
        total_levels = 10
        for level in range(total_levels):
            total_sum_pos = int(1.0 * level * sorted_hit_cum.shape[0] / total_levels)
            total_sum = torch.sum(sorted_hit_cum[:total_sum_pos])
            levels.append(total_sum / total_hit)
            x.append(total_sum_pos / (csr_topo.indptr.shape[0] - 1))
        print(levels)

        plt.plot(x, levels, label=f"partition_num={partition_num}")
        plt.scatter(x, levels, label=f"partition_num={partition_num}")
    plt.savefig("lj_30_cdf.png")


def test_random_partiton_without_replication():
    print(f"{'=' * 30 } Random Partition {'=' * 30 }")
    train_idx, csr_topo, quiver_sampler = load_products()
    

    for partition_num in range(1, 10, 1):
        idx_len = train_idx.size(0)
        random.shuffle(train_idx)
        partition_book = torch.randint(0, partition_num, size = (csr_topo.node_count, ))

        distribution = [((partition_book == partition).nonzero()).shape[0] for partition in range(partition_num)]

        print(f"partition distribution: {distribution}")


        local_train_idx = train_idx[:idx_len // partition_num]

        train_loader = torch.utils.data.DataLoader(local_train_idx,
                                                batch_size=1024,
                                                pin_memory=True,
                                                shuffle=True)
        total_count = 0
        total_hit = 0
        for _ in range(10):
            for seeds in train_loader:
                n_id, _, _ = quiver_sampler.sample(seeds)
                
                hit_count = ((partition_book[n_id] == 0).nonzero()).shape[0]
                total_hit += hit_count
                total_count += n_id.shape[0]
        print(f"Partition = {partition_num}, Local hit rate = {total_hit / total_count}")



def test_random_partition_with_hot_replicate():
    print(f"{'=' * 30 } Random Partition With Replication {'=' * 30 }")
    train_idx, csr_topo, quiver_sampler = load_reddit()

    cache_rate = 0.2
    node_degree = csr_topo.degree
    _, idx = torch.sort(node_degree, descending=True)
    x = []
    y = []

    for partition_num in range(2, 10, 1):
        idx_len = train_idx.size(0)
        random.shuffle(train_idx)
        feature = torch.arange(0, csr_topo.node_count, dtype=torch.long)
        feature = feature[idx]
        cached_nodes = int(cache_rate * csr_topo.node_count)
        # random partition
        partition_book = torch.randint(0, partition_num, size = (csr_topo.node_count, ))
        # cache hot data 
        partition_book[idx[:cached_nodes]] = 0

        distribution = [((partition_book == partition).nonzero()).shape[0] for partition in range(partition_num)]

        print(f"partition distribution: {distribution}")


        local_train_idx = train_idx[:idx_len // partition_num]

        train_loader = torch.utils.data.DataLoader(local_train_idx,
                                                batch_size=1024,
                                                pin_memory=True,
                                                shuffle=True)
        total_count = 0
        total_hit = 0
        for _ in range(10):
            for seeds in train_loader:
                n_id, _, _ = quiver_sampler.sample(seeds)
                hit_count = ((partition_book[n_id] == 0).nonzero()).shape[0]
                total_hit += hit_count
                total_count += n_id.shape[0]
        print(f"Partition = {partition_num}, Local hit rate = {total_hit / total_count}")
        x.append(partition_num)
        y.append(total_hit / total_count)

    plt.plot(x, y)


def test_quiver_partition_without_replication():
    print(f"{'=' * 30 } Quiver Partition {'=' * 30 }")
    train_idx, csr_topo, quiver_sampler = load_reddit()
    torch.cuda.set_device(0)
    x = []
    y = []
    

    for partition_num in range(2, 10, 1):
        idx_len = train_idx.size(0)
        random.shuffle(train_idx)
        probs = []
        for partition in range(partition_num):
            partition_train_idx = train_idx[(idx_len // partition_num) * partition: (idx_len // partition_num) * (partition+1)]
            prob = quiver_sampler.sample_prob(partition_train_idx, csr_topo.node_count)
            probs.append(prob)

        partition_res, cache_res = quiver.quiver_partition(probs, "partition_result_dir")
        
        local_train_idx = train_idx[:idx_len // partition_num]

        train_loader = torch.utils.data.DataLoader(local_train_idx,
                                                batch_size=1024,
                                                pin_memory=True,
                                                shuffle=True)
        
        partition_book = torch.zeros(csr_topo.node_count, dtype=torch.int32)
        partition_book[:] = partition_num

        partition_book[partition_res[0]] = 0
        if cache_res[0] is not None:
            partition_book[cache_res[0]] = 0

        total_count = 0
        total_hit = 0
        for _ in range(10):
            for seeds in train_loader:
                n_id, _, _ = quiver_sampler.sample(seeds)
                hit_count = ((partition_book[n_id] == 0).nonzero()).shape[0]
                total_hit += hit_count
                total_count += n_id.shape[0]
        print(f"Partition = {partition_num}, Local hit rate = {total_hit / total_count}")
        x.append(partition_num)
        y.append(total_hit / total_count)
    
    plt.plot(x, y)

def test_load_partition():
    result_path = "partition_result_dir"
    partition_res, cache_res = quiver.load_quiver_partition(0, result_path)
    print(partition_res)
    print(cache_res)

# Uncomment to check CDF of a certain dataset
#test_cdf()

# Uncomment to test random partition
#test_random_partiton_without_replication()

# Uncomment to test random partition with degree based replication
#test_random_partition_with_hot_replicate()

# Uncomment to test quiver partition algorithm
test_quiver_partition_without_replication()

# Uncomment to test partition loading
test_load_partition()

# Draw figure
plt.savefig("local_hit_rate.png")
