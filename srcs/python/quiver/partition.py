import torch

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
