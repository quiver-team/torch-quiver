import torch
from quiver.utils import cal_memory_budget_to_bytes

QUIVER_MAGIC_NUMBER = 128

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

    res = [None] * partitioned_num
    for rank in range(partitioned_num):
        res[rank] = []

    chunk_size = QUIVER_MAGIC_NUMBER * partitioned_num
    chunk_num = (total_node_num + chunk_size - 1) // chunk_size

    current_chunk_start_pos = 0
    current_partition_idx = 0
    for i in range(chunk_num):
        current_chunk_end_pos = min(total_node_num, current_chunk_start_pos + chunk_size)
        current_chunk_size = current_chunk_end_pos - current_chunk_start_pos
        chunk = torch.arange(current_chunk_start_pos,
                             current_chunk_end_pos,
                             dtype=torch.int64,
                             device=device)
        probs_sum_chunk = [
            torch.zeros(current_chunk_size, device=device) + 1e-6 for i in range(partitioned_num)
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
            res[rank].append(pick_ids)
            probs_sum_chunk[partition_idx][pick_chunk_part] = -1
            assigned_node_size += actual_per_partition_size
        current_partition_idx += 1
        current_chunk_start_pos += current_chunk_size

    for partition_idx in range(partitioned_num):
        res[partition_idx] = torch.cat(res[partition_idx])
    return res


def partition_with_replication(device, probs, ids, per_rank_size):
    """Partition node with given node IDs and node access distribution.
    The result will cause replication between each parititon,
    but the size of each partition will not exceed per_rank_size.
    """
    partition_res = partition_without_replication(device, probs, ids)
    if ids is not None:
        ids = ids.to(device)
    ranks = len(probs)
    total_res = [
        torch.empty(per_rank_size, device=device) for i in range(ranks)
    ]
    probs = [prob.clone().to(device) for prob in probs]
    for rank in range(ranks):
        partition_ids = partition_res[rank]
        probs[rank][partition_ids] = -1e6
        replication_size = per_rank_size - partition_ids.size(0)
        _, prev_order = torch.sort(probs[rank], descending=True)
        replication_ids = ids[
            prev_order[:
                       replication_size]] if ids is not None else prev_order[:
                                                                             replication_size]
        total_res[rank] = torch.cat((partition_ids, replication_ids))
    return total_res


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


def partition_free(device, probs, ids, per_rank_size):
    """Partition node with given node IDs and node access distribution.
    The result will cause either replication or missing nodes across partitions.
    The size of each partition is limited by per_rank_size.
    """
    prob_sum, node_ids = select_nodes(device, probs, ids)
    nodes = node_ids.size(0)
    ranks = len(probs)
    limit = ranks * per_rank_size
    if nodes <= limit:
        return partition_with_replication(device, probs, node_ids,
                                          per_rank_size), None
    else:
        _, prev_order = torch.sort(prob_sum, descending=True)
        limit_ids = prev_order[:limit]
        return partition_without_replication(device, probs,
                                             node_ids), limit_ids


def partition(probs, result_path, cache_memory_budget=0, per_feature_size=0):
    """
    Partition graph topology based on access probability

    Args:
        result_path (str): path for partition result
        cache_memory_budget (Union[str, int, float]): user-specified memory budget for caching hot feature
        per_feature_size (Union[str, int, float]): per-feature size for user's feature
    
    Returns:
        None
    """
    cache_memory_budget_bytes = cal_memory_budget_to_bytes(cache_memory_budget)
    per_feature_size_bytes = cal_memory_budget_to_bytes(per_feature_size)

    current_device = torch.cuda.current_device()


