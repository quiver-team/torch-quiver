import torch
import shutil
import os
from typing import List
import quiver.utils as quiver_util

__all__ = [
    "quiver_partition_feature", "load_quiver_feature_partition",
    "partition_without_replication", "select_nodes"
]

QUIVER_MAGIC_NUMBER = 256
CHUNK_NUM = 32


def partition_without_replication(device, probs, ids):
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


def select_nodes(device, probs, ids):
    nodes = probs[0].size(0)
    prob_sum = torch.zeros(nodes, device=device)
    for prob in probs:
        if ids is None:
            prob_sum += prob
        else:
            prob_sum[ids] += prob[ids]
    node_ids = torch.nonzero(prob_sum)
    return prob_sum, node_ids


def partition_feature_without_replication(probs: List[torch.Tensor],
                                          chunk_size: int):
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
    chunk_num = (total_node_num + chunk_size - 1) // chunk_size

    current_chunk_start_pos = 0
    current_partition_idx = 0
    for _ in range(chunk_num):
        current_chunk_end_pos = min(total_node_num,
                                    current_chunk_start_pos + blob_size)
        current_chunk_size = current_chunk_end_pos - current_chunk_start_pos
        chunk = torch.arange(current_chunk_start_pos,
                             current_chunk_end_pos,
                             device=device)
        probs_sum_chunk = [
            torch.zeros(current_chunk_size, device=device) + 1e-6
            for _ in range(partitioned_num)
        ]
        for src_rank in range(partitioned_num):
            for dst_rank in range(partitioned_num):
                if dst_rank == src_rank:
                    probs_sum_chunk[
                        src_rank] += probs[dst_rank][chunk] * partitioned_num
                else:
                    probs_sum_chunk[src_rank] -= probs[dst_rank][chunk]
        assigned_node_size = 0
        per_partition_size = chunk_size
        for partition_idx in range(current_partition_idx,
                                   current_partition_idx + partitioned_num):
            partition_idx = partition_idx % partitioned_num
            actual_per_partition_size = min(
                per_partition_size, current_chunk_size - assigned_node_size)
            _, sorted_res_order = torch.sort(probs_sum_chunk[partition_idx],
                                             descending=True)
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


def quiver_partition_feature(probs: torch.Tensor,
                             result_path: str,
                             cache_memory_budget=0,
                             per_feature_size=0,
                             chunk_size=QUIVER_MAGIC_NUMBER):
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
        res = input(
            f"{result_path} already exists, enter Y/N to continue, If continue, {result_path} will be deleted:"
        )
        res = res.upper()
        if res == "Y":
            shutil.rmtree(result_path)
        else:
            print("exiting ...")
            exit()

    partition_num = len(probs)

    # create result folder
    for partition_idx in range(partition_num):
        os.makedirs(
            os.path.join(result_path, f"feature_partition_{partition_idx}"))

    # calculate cached feature count
    cache_memory_budget_bytes = quiver_util.parse_size(cache_memory_budget)
    per_feature_size_bytes = quiver_util.parse_size(per_feature_size)
    cache_count = int(cache_memory_budget_bytes /
                      (per_feature_size_bytes + 1e-6))
    per_partition_cache_count = cache_count // partition_num

    partition_book = torch.zeros(probs[0].shape,
                                 dtype=torch.int64,
                                 device=torch.cuda.current_device())
    partition_res, changed_probs = partition_feature_without_replication(
        probs, chunk_size)

    cache_res = [None] * partition_num

    if cache_count > 0:
        for partition_idx in range(partition_num):
            _, prev_order = torch.sort(changed_probs[partition_idx],
                                       descending=True)
            cache_res[partition_idx] = prev_order[:per_partition_cache_count]

    for partition_idx in range(partition_num):
        partition_result_path = os.path.join(
            result_path, f"feature_partition_{partition_idx}",
            "partition_res.pth")
        cache_result_path = os.path.join(result_path,
                                         f"feature_partition_{partition_idx}",
                                         "cache_res.pth")
        partition_book[partition_res[partition_idx]] = partition_idx
        torch.save(partition_res[partition_idx], partition_result_path)
        torch.save(cache_res[partition_idx], cache_result_path)

    partition_book_path = os.path.join(result_path,
                                       f"feature_partition_book.pth")
    torch.save(partition_book, partition_book_path)

    return partition_book, partition_res, cache_res


def load_quiver_feature_partition(partition_idx: int, result_path: str):
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

    partition_result_path = os.path.join(result_path,
                                         f"feature_partition_{partition_idx}",
                                         "partition_res.pth")
    cache_result_path = os.path.join(result_path,
                                     f"feature_partition_{partition_idx}",
                                     "cache_res.pth")
    partition_book_path = os.path.join(result_path,
                                       f"feature_partition_book.pth")

    partition_book = torch.load(partition_book_path)
    partition_res = torch.load(partition_result_path)
    cache_res = torch.load(cache_result_path)

    return partition_book, partition_res, cache_res
