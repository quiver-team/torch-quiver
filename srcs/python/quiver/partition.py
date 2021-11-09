import torch
import time

CHUNK_NUM = 24


def partition_without_replication(device, probs, ids):
    t0 = time.time()
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
    t1 = time.time()
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
    t2 = time.time()
    for rank in range(ranks):
        res[rank] = torch.cat(res[rank])
        if ids is not None:
            res[rank] = ids[res[rank]]
    t3 = time.time()
    return res


def partition_with_replication(device, probs, ids, per_rank_size):
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