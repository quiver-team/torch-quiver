import torch


import quiver
from ogb.lsc import MAG240MDataset
import os.path as osp
import random
from ogb.nodeproppred import PygNodePropPredDataset
from matplotlib import pyplot as plt
from torch_geometric.datasets import Reddit

import torch
import shutil
import os
from typing import List
import quiver.utils as quiver_util



__all__ = ["quiver_partition_feature", "load_quiver_feature_partition"]

torch.manual_seed(0)

QUIVER_MAGIC_NUMBER = 1024

def partition_feature_without_replication(probs: List[torch.Tensor], chunk_size: int):
    """Partition node with node access distribution. 
    The result will cause no replication between each parititon.

    Args:
        probs (torch.Tensor): node access distribution
        chunk_size (int): chunk_size

    Returns:
        [torch.Tensor]: list of IDs for each partition

    """

    device = torch.cuda.current_device()
    partitioned_num = len(probs)

    probs = [prob.to(device) for prob in probs]
    total_node_num = probs[0].size(0)

    res = [[] for _ in range(partitioned_num)]

    blob_size = chunk_size * partitioned_num
    chunk_num = (total_node_num + blob_size - 1) // blob_size

    current_chunk_start_pos = 0
    current_partition_idx = partitioned_num - 1
    for _ in range(chunk_num):
        current_chunk_end_pos = min(total_node_num, current_chunk_start_pos + blob_size)
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
        per_partition_size = (current_chunk_size  + partitioned_num - 1) // partitioned_num 
        for partition_idx in range(current_partition_idx, current_partition_idx + partitioned_num):
            partition_idx = partition_idx % partitioned_num
            actual_per_partition_size = min(per_partition_size, current_chunk_size - assigned_node_size)
            _, sorted_res_order = torch.sort(probs_sum_chunk[partition_idx], descending=True)
            pick_chunk_part = sorted_res_order[:actual_per_partition_size]
            pick_ids = chunk[pick_chunk_part]
            res[partition_idx].append(pick_ids)
            for idx in range(partitioned_num):
                probs_sum_chunk[idx][pick_chunk_part] = -1
            assigned_node_size += actual_per_partition_size
        current_partition_idx += 1
        current_chunk_start_pos += current_chunk_size

    for partition_idx in range(partitioned_num):
        res[partition_idx] = torch.cat(res[partition_idx])
    return res, probs

def quiver_partition_feature(probs:torch.Tensor, result_path: str, cache_memory_budget=0, per_feature_size=0, chunk_size=QUIVER_MAGIC_NUMBER):
    """
    Partition graph feature based on access probability and generate result folder. The final result folder will be like:
    
    -result_path
        -partition_0
            -partition_res.pth
            -cache_res.pth
        -partition_1
            -partition_res.pth
            -cache_res.pth
        -partition_2
            -partition_res.pth
            -cache_res.pth
        ...

    Args:
        probs:
        result_path (str): path for partition result
        cache_memory_budget (Union[str, int, float]): user-specified memory budget for caching hot feature
        per_feature_size (Union[str, int, float]): per-feature size for user's feature
    
    Returns:
        partition_book (torch.Tensor): Indicates which partition_idx a node belongs to
        feature_partition_res (torch.Tensor): partitioned feature result
        feature_cache_res (torch.Tensor): cached feature result
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
        os.makedirs(os.path.join(result_path, f"feature_partition_{partition_idx}"))
    
    # calculate cached feature count
    cache_memory_budget_bytes = quiver_util.parse_size(cache_memory_budget)
    per_feature_size_bytes = quiver_util.parse_size(per_feature_size)
    cache_count = int(cache_memory_budget_bytes / (per_feature_size_bytes + 1e-6))
    per_partition_cache_count = cache_count

    partition_book = torch.zeros(probs[0].shape, dtype=torch.int64, device=torch.cuda.current_device())
    partition_res, changed_probs = partition_feature_without_replication(probs, chunk_size)
    
    cache_res = [None] * partition_num

    if cache_count > 0:
        for partition_idx in range(partition_num):
            changed_probs[partition_idx][partition_res[partition_idx]] = -1e3
            _, prev_order = torch.sort(changed_probs[partition_idx], descending=True)
            cache_res[partition_idx] = prev_order[: per_partition_cache_count]
    
    for partition_idx in range(partition_num):
        partition_result_path = os.path.join(result_path, f"feature_partition_{partition_idx}", "partition_res.pth")
        cache_result_path = os.path.join(result_path, f"feature_partition_{partition_idx}", "cache_res.pth")
        partition_book[partition_res[partition_idx]] = partition_idx
        torch.save(partition_res[partition_idx], partition_result_path)
        torch.save(cache_res[partition_idx], cache_result_path)
    
    partition_book_path = os.path.join(result_path, f"feature_partition_book.pth")
    torch.save(partition_book, partition_book_path)

    return partition_book, partition_res, cache_res



def load_quiver_feature_partition(partition_idx: int, result_path:str):
    """
    Load partition result for partition ${partition_idx}

    Args:
        partition_idx (int): Partition idx
        partition_result_path (str): partition result path
    
    Returns:
        partition_book (torch.Tensor): partition_book indicates which partition_idx a node belongs to
        partition_res (torch.Tensor): node indexes belong to this partition
        cache_res (torch.Tensor): cached node indexes belong to this partition

    """

    if not os.path.exists(result_path):
        raise Exception("Result path not exists")
    
    partition_result_path = os.path.join(result_path, f"feature_partition_{partition_idx}", "partition_res.pth")
    cache_result_path = os.path.join(result_path, f"feature_partition_{partition_idx}", "cache_res.pth")
    partition_book_path = os.path.join(result_path, f"feature_partition_book.pth")
    

    partition_book = torch.load(partition_book_path)
    partition_res = torch.load(partition_result_path)
    cache_res = torch.load(cache_result_path)

    return partition_book, partition_res, cache_res


CHUNK_NUM = 32
def old_partition_without_replication(probs, device = 0, ids=None):
    """Partition node with given node IDs and node access distribution.
    The result will cause no replication between each parititon.
    We assume node IDs can be placed in the given device.
    Args:
        device (int): device which computes the partitioning strategy
        probs (torch.Tensor): node access distribution
        ids (Optional[torch.Tensor]): specified node IDs
    Returns:
        [torch.Tensor]: list of IDs for each partition
    """
    ranks = len(probs)
    if ids is not None:
        ids = ids.to(device)
    probs = [
        prob[ids].to(device) if ids is not None else prob.to(device)
        for prob in probs
    ]
    total_size = ids.size(0) if ids is not None else probs[0].size(0)
    res = [None] * ranks
    for rank in range(ranks):
        res[rank] = []
    CHUNK_SIZE = (total_size + CHUNK_NUM - 1) // CHUNK_NUM
    chunk_beg = 0
    beg_rank = 0
    for i in range(CHUNK_NUM):
        chunk_end = min(total_size, chunk_beg + CHUNK_SIZE)
        chunk_size = chunk_end - chunk_beg
        chunk = torch.arange(chunk_beg,
                             chunk_end,
                             dtype=torch.int64,
                             device=device)
        probs_sum_chunk = [
            torch.zeros(chunk_size, device=device) + 1e-6 for i in range(ranks)
        ]
        for rank in range(ranks):
            for dst_rank in range(ranks):
                if dst_rank == rank:
                    probs_sum_chunk[rank] += probs[dst_rank][chunk] * ranks
                else:
                    probs_sum_chunk[rank] -= probs[dst_rank][chunk]
        acc_size = 0
        rank_size = (chunk_size + ranks - 1) // ranks
        picked_chunk_parts = torch.LongTensor([]).to(device)
        for rank_ in range(beg_rank, beg_rank + ranks):
            rank = rank_ % ranks
            probs_sum_chunk[rank][picked_chunk_parts] -= 1e6
            rank_size = min(rank_size, chunk_size - acc_size)
            _, rank_order = torch.sort(probs_sum_chunk[rank], descending=True)
            pick_chunk_part = rank_order[:rank_size]
            pick_ids = chunk[pick_chunk_part]
            picked_chunk_parts = torch.cat(
                (picked_chunk_parts, pick_chunk_part))
            res[rank].append(pick_ids)
            acc_size += rank_size
        beg_rank += 1
        chunk_beg += chunk_size
    for rank in range(ranks):
        res[rank] = torch.cat(res[rank])
        if ids is not None:
            res[rank] = ids[res[rank]]
    return res


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
    indptr = torch.load("/data/papers/ogbn_papers100M/csr/indptr.pt")
    indices = torch.load("/data/papers/ogbn_papers100M/csr/indices.pt")
    train_idx = torch.load("/data/papers/ogbn_papers100M/index/train_idx.pt")
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [1], 0, mode="UVA")
    print(f"average degree of paper100M = {torch.sum(csr_topo.degree) / csr_topo.node_count}")
    return train_idx, csr_topo, quiver_sampler

def load_products():
    root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15], 0, mode="UVA")
    print(f"average degree of products = {torch.sum(csr_topo.degree) / csr_topo.node_count}")
    return train_idx, csr_topo, quiver_sampler

def load_reddit():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [10, 7, 3], 0, mode="UVA")
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
    train_idx, csr_topo, quiver_sampler = load_reddit()
    
    for partition_num in range(4, 5, 1):
        idx_len = train_idx.size(0)
        # random.shuffle(train_idx)
        train_idx = train_idx[torch.randperm(idx_len)]
        global_partition_book = torch.randint(0, partition_num, size = (csr_topo.node_count, ))


        local_partition_books = []
        for partition in range(partition_num):
            local_partition_book = global_partition_book.clone()
            if partition == partition_num - 1:
                local_partition_book[train_idx[(idx_len // partition_num) * partition:]] = partition
                local_train_idx = train_idx[(idx_len // partition_num) * partition:]
            else:
                local_partition_book[train_idx[(idx_len // partition_num) * partition: (idx_len // partition_num) * (partition+1)]] = partition
                local_train_idx = train_idx[(idx_len // partition_num) * partition: (idx_len // partition_num) * (partition+1)]
            local_partition_books.append(local_partition_book)

            train_loader = torch.utils.data.DataLoader(local_train_idx,
                                                    batch_size=1024,
                                                    pin_memory=True,
                                                    shuffle=True)
            total_count = 0
            total_hit = 0
            for _ in range(2):
                for seeds in train_loader:
                    n_id, _, _ = quiver_sampler.sample(seeds)
                    hit_count = ((local_partition_books[partition][n_id] == partition).nonzero()).shape[0]
                    total_hit += hit_count
                    total_count += n_id.shape[0]
            print(f"Partition = {partition_num}, Local hit rate = {total_hit / total_count}")



def test_random_partition_with_hot_replicate():
    print(f"{'=' * 30 } Random Partition With Replication {'=' * 30 }")
    train_idx, csr_topo, quiver_sampler = load_products()

    cache_rate = 0.2
    node_degree = csr_topo.degree
    _, idx = torch.sort(node_degree, descending=True)
    x = []
    y = []

    hit_rate_results = []
    for partition_num in range(4, 5, 1):
        if partition_num > 1:
            cache_rate += cache_rate * (1/(partition_num-1))
        
        idx_len = train_idx.size(0)
        # random.shuffle(train_idx)
        train_idx = train_idx[torch.randperm(idx_len)]
        feature = torch.arange(0, csr_topo.node_count, dtype=torch.long)
        feature = feature[idx]
        cached_nodes = int(cache_rate * csr_topo.node_count)
        # random partition
        global_partition_book = torch.randint(0, partition_num, size = (csr_topo.node_count, ))

        # cache hot data 
        local_partition_books = []
        for partition in range(partition_num):
            local_partition_book = global_partition_book.clone()
            local_partition_book[idx[:cached_nodes]] = partition
            local_partition_books.append(local_partition_book)


        for partition in range(partition_num):
            if partition == partition_num - 1:
                local_train_idx = train_idx[(idx_len // partition_num) * partition: ]
            else:
                local_train_idx = train_idx[(idx_len // partition_num) * partition: (idx_len // partition_num) * (partition+1)]


            train_loader = torch.utils.data.DataLoader(local_train_idx,
                                                    batch_size=1024,
                                                    pin_memory=True,
                                                    shuffle=True)
            total_count = 0
            total_hit = 0

            for _ in range(2):
                for seeds in train_loader:
                    n_id, _, _ = quiver_sampler.sample(seeds)
                    hit_count = ((local_partition_books[partition][n_id] == partition).nonzero()).shape[0]
                    total_hit += hit_count
                    total_count += n_id.shape[0]
            print(f"Partition = {partition}, Local hit rate = {total_hit / total_count}")
            x.append(partition)
            y.append(total_hit / total_count)

    plt.plot(x, y)



def test_quiver_partition_without_replication():
    print(f"{'=' * 30 } Quiver Partition {'=' * 30 }")
    train_idx, csr_topo, quiver_sampler = load_products()
    torch.cuda.set_device(0)

    cache_rate = 0.2
    

    for partition_num in range(4, 5, 1):
        idx_len = train_idx.size(0)
        shuffled_train_idx = train_idx[torch.randperm(idx_len)]
        partition_train_idx_list = []
        probs = []
        for partition in range(partition_num):
            if partition == partition_num - 1:
                partition_train_idx = shuffled_train_idx[(idx_len // partition_num) * partition: ]
            else:
                partition_train_idx = shuffled_train_idx[(idx_len // partition_num) * partition: (idx_len // partition_num) * (partition + 1)]
            partition_train_idx_list.append(partition_train_idx)
            prob = quiver_sampler.sample_prob(partition_train_idx, csr_topo.node_count)             
            probs.append(prob)

        prob_sums = [torch.sum(probs[idx]) for idx in range(partition_num)]
        print(f"Check probs sum result: {prob_sums}")

        partition_book, partition_res, cache_res = quiver_partition_feature(probs, "partition_result_dir", f"{int(cache_rate * csr_topo.node_count)}KB", "1KB")
        

        for partition_idx in range(partition_num):
            print(f"For partition_{partition_idx}: partitioned node count = {partition_res[partition_idx].shape[0]}")

        for partition in range(partition_num):

            local_partition_book = partition_book.clone()

            train_loader = torch.utils.data.DataLoader(partition_train_idx_list[partition],
                                                    batch_size=1024,
                                                    pin_memory=True,
                                                    shuffle=True)
        
        
            if cache_res[partition] is not None:
                local_partition_book[cache_res[partition]] = partition

            total_count = 0
            total_hit = 0
            for seeds in train_loader:
                n_id, _, _ = quiver_sampler.sample(seeds)
                hit_count = ((local_partition_book[n_id] == partition).nonzero()).shape[0]
                total_hit += hit_count
                total_count += n_id.shape[0]
            print(f"PartitionNum = {partition_num}, PartitionIdx = {partition}, Local hit rate = {total_hit / total_count}")


def test_load_partition():
    result_path = "partition_result_dir"
    partition_book, partition_res, cache_res = quiver.load_quiver_feature_partition(0, result_path)
    print(partition_res)
    print(cache_res)

def test_old_partition():
    print(f"{'=' * 30 } Quiver Old Partition {'=' * 30 }")
    train_idx, csr_topo, quiver_sampler = load_com_lj()
    torch.cuda.set_device(0)

    

    for partition_num in range(4, 5, 1):
        idx_len = train_idx.size(0)
        random.shuffle(train_idx)
        probs = []
        for partition in range(partition_num):
            partition_train_idx = train_idx[(idx_len // partition_num) * partition: (idx_len // partition_num) * (partition+1)]
            prob = quiver_sampler.sample_prob(partition_train_idx, csr_topo.node_count)
            probs.append(prob)

        partition_res = old_partition_without_replication(probs)
        
        partition_book = torch.randint(0, partition_num, size = (csr_topo.node_count, ))
        partition_book[:] = partition_num

        for partition_idx in range(partition_num):
            print(f"For partition_{partition_idx}: partitioned node count = {partition_res[partition_idx].shape[0]}")
            partition_book[partition_res[partition_idx]] = partition_idx

        for partition in range(partition_num):

            local_partition_book = partition_book.clone()
            local_train_idx = train_idx[(idx_len // partition_num) * partition: (idx_len // partition_num) * (partition+1)]


            train_loader = torch.utils.data.DataLoader(local_train_idx,
                                                    batch_size=1024,
                                                    pin_memory=True,
                                                    shuffle=True)
        


            total_count = 0
            total_hit = 0
            for seeds in train_loader:
                n_id, _, _ = quiver_sampler.sample(seeds)
                hit_count = ((local_partition_book[n_id] == partition).nonzero()).shape[0]
                total_hit += hit_count
                total_count += n_id.shape[0]
            print(f"PartitionNum = {partition_num}, PartitionIdx = {partition}, Local hit rate = {total_hit / total_count}")


# Uncomment to check CDF of a certain dataset
#test_cdf()

# Uncomment to test random partition
test_random_partiton_without_replication()

# Uncomment to test random partition with degree based replication
# test_random_partition_with_hot_replicate()

# Uncomment to test old partition
#test_old_partition()
# Uncomment to test quiver partition algorithm
# test_quiver_partition_without_replication()

# Uncomment to test partition loading
#test_load_partition()

# Draw figure
plt.savefig("local_hit_rate.png")
