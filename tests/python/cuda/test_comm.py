import torch
import torch.multiprocessing as mp
import torch_quiver as torch_qv
import quiver
import torch.distributed as dist
import os
import time

LOCAL_ADDR = '192.168.0.78'
MASTER_ADDR = '192.168.0.78'
MASTER_PORT = 12355


def child_sendrecv_proc(rank, ws, id):
    torch.cuda.set_device(rank)
    comm = torch_qv.NcclComm(rank, ws, id)
    print(f"{rank} ready")
    if rank == 0:
        a = torch.zeros(10, device=0)
        comm.send(a, 1)
    else:
        a = torch.ones(10, device=1)
        comm.recv(a, 0)
    print(f"{rank} tensor {a}")
    num = 10000
    size = 1024 * 1024
    large = torch.zeros(size, device=rank)
    for i in range(5):
        if rank == 0:
            comm.send(large, 1)
        else:
            comm.recv(large, 0)
        torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(num):
        if rank == 0:
            comm.send(large, 1)
        else:
            comm.recv(large, 0)
        torch.cuda.current_stream().synchronize()
    t1 = time.time()
    if rank == 1:
        print(f"Latency {(t1 - t0) / num}")
        print(f"Throughput {num * size * 4 / 1024 / 1024 / 1024 / (t1 - t0)}")


def child_sendrecv_proc_pair(rank, ws, id):
    torch.cuda.set_device(rank // 2)
    comm = torch_qv.NcclComm(rank, ws, id)
    print(f"{rank} ready")
    if rank % 2 == 0:
        a = torch.zeros(10, device=rank // 2)
        comm.send(a, rank + 1)
    else:
        a = torch.ones(10, device=rank // 2)
        comm.recv(a, rank - 1)
    torch.cuda.current_stream().synchronize()
    print(f"{rank} tensor {a}")
    num = 10000
    size = 1024 * 1024
    large = torch.zeros(size, device=rank // 2)
    for i in range(5):
        if rank % 2 == 0:
            comm.send(large, rank + 1)
        else:
            comm.recv(large, rank - 1)
        torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(num):
        if rank % 2 == 0:
            comm.send(large, rank + 1)
        else:
            comm.recv(large, rank - 1)
        torch.cuda.current_stream().synchronize()
    t1 = time.time()
    if rank % 2 == 1:
        print(f"Latency {(t1 - t0) / num}")
        print(f"Throughput {num * size * 4 / 1024 / 1024 / 1024 / (t1 - t0)}")


def child_sendrecv_proc_pair_bidirect(rank, ws, id):
    torch.cuda.set_device(rank // 2)
    comm = torch_qv.NcclComm(rank, ws, id)
    print(f"{rank} ready")
    if rank % 2 == 0:
        a = torch.zeros(10, device=rank // 2)
        comm.send(a, rank + 1)
    else:
        a = torch.ones(10, device=rank // 2)
        comm.recv(a, rank - 1)
    torch.cuda.current_stream().synchronize()
    print(f"{rank} tensor {a}")
    num = 10000
    size = 1024 * 1024
    large = torch.zeros(size, device=rank // 2)
    large_inverse = torch.zeros(size, device=rank // 2)
    stream = torch.cuda.Stream()
    stream_inverse = torch.cuda.Stream()
    for i in range(5):
        if rank // 2 == 0:
            if rank % 2 == 0:
                comm.send(large, rank + 1)
            else:
                comm.recv(large, rank - 1)
        else:
            if rank % 2 == 0:
                comm.recv(large, rank + 1)
            else:
                comm.send(large, rank - 1)
        torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(num):
        if rank // 2 == 0:
            if rank % 2 == 0:
                comm.send(large, rank + 1)
            else:
                comm.recv(large, rank - 1)
        else:
            if rank % 2 == 0:
                comm.recv(large_inverse, rank + 1)
            else:
                comm.send(large_inverse, rank - 1)
        torch.cuda.current_stream().synchronize()
    t1 = time.time()
    if rank % 2 == 1:
        print(f"Latency {(t1 - t0) / num}")
        print(f"Throughput {num * size * 4 / 1024 / 1024 / 1024 / (t1 - t0)}")


def child_allreduce_proc(rank, ws, id):
    torch.cuda.set_device(rank)
    comm = torch_qv.NcclComm(rank, ws, id)
    print(f"{rank} ready")
    if rank == 0:
        a = torch.zeros(10, device=0)
    else:
        a = torch.ones(10, device=1)
    comm.allreduce(a)
    print(f"{rank} tensor {a}")
    num = 100
    size = 1024 * 1024
    large = torch.zeros(size, device=rank)
    for i in range(5):
        comm.allreduce(large)
        torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(num):
        comm.allreduce(large)
        torch.cuda.current_stream().synchronize()
    t1 = time.time()
    if rank == 1:
        print(f"Latency {(t1 - t0) / num}")
        print(f"Throughput {num * size * 4 / 1024 / 1024 / 1024 / (t1 - t0)}")


def child_allreduce_proc_pair(rank, ws, id):
    torch.cuda.set_device(rank // 2)
    comm = torch_qv.NcclComm(rank, ws, id)
    print(f"{rank} ready")
    if rank % 2 == 0:
        a = torch.zeros(10, device=rank // 2)
    else:
        a = torch.ones(10, device=rank // 2)
    comm.allreduce(a)
    torch.cuda.current_stream().synchronize()
    print(f"{rank} tensor {a}")
    num = 10
    size = 1024 * 1024 * 1024
    large = torch.zeros(size, device=rank // 2)
    for i in range(5):
        comm.allreduce(large)
        torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(num):
        comm.allreduce(large)
        torch.cuda.current_stream().synchronize()
    t1 = time.time()
    if rank == 1:
        print(f"Latency {(t1 - t0) / num}")
        print(f"Throughput {num * size * 4 / 1024 / 1024 / 1024 / (t1 - t0)}")


def test_local():
    id = quiver.comm.getNcclId()
    ws = 2
    procs = []
    for i in range(ws):
        proc = mp.Process(target=child_sendrecv_proc, args=(i, ws, id))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()


def test_dist(rank):
    ws = 2
    store = dist.TCPStore(MASTER_ADDR, MASTER_PORT, ws,
                          MASTER_ADDR == LOCAL_ADDR)
    if rank == 0:
        id = quiver.comm.getNcclId()
        store.set("id", id)
    else:
        id = store.get("id")
    print(f"{rank} init store {id}")
    child_sendrecv_proc(rank, ws, id)


def test_dist_pair(rank):
    local_size = 2
    store = dist.TCPStore(MASTER_ADDR, MASTER_PORT, 2,
                          MASTER_ADDR == LOCAL_ADDR)
    if rank == 0:
        id = quiver.comm.getNcclId()
        store.set("id", id)
    else:
        id = store.get("id")
    print(f"{rank} init store {id}")
    procs = []
    for i in range(local_size):
        proc = mp.Process(target=child_sendrecv_proc_pair,
                          args=(i * 2 + rank, 2 * local_size, id))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()


def test_dist_pair_bidirect(rank):
    local_size = 2
    store = dist.TCPStore(MASTER_ADDR, MASTER_PORT, 2,
                          MASTER_ADDR == LOCAL_ADDR)
    if rank == 0:
        id = quiver.comm.getNcclId()
        store.set("id", id)
    else:
        id = store.get("id")
    print(f"{rank} init store {id}")
    procs = []
    for i in range(local_size):
        proc = mp.Process(target=child_sendrecv_proc_pair_bidirect,
                          args=(i * 2 + rank, 2 * local_size, id))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()


def test_nccl_allreduce(rank):
    ws = 2
    store = dist.TCPStore(MASTER_ADDR, MASTER_PORT, ws,
                          MASTER_ADDR == LOCAL_ADDR)
    if rank == 0:
        id = quiver.comm.getNcclId()
        store.set("id", id)
    else:
        id = store.get("id")
    print(f"{rank} init store {id}")
    child_allreduce_proc(rank, ws, id)


def test_nccl_allreduce_pair(rank):
    local_size = 2
    store = dist.TCPStore(MASTER_ADDR, MASTER_PORT, 2,
                          MASTER_ADDR == LOCAL_ADDR)
    if rank == 0:
        id = quiver.comm.getNcclId()
        store.set("id", id)
    else:
        id = store.get("id")
    print(f"{rank} init store {id}")
    procs = []
    for i in range(local_size):
        proc = mp.Process(target=child_allreduce_proc_pair,
                          args=(i * 2 + rank, 2 * local_size, id))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()


def child_feat_partition(rank, ws, id, device, host, hosts, global2host):
    torch.cuda.set_device(device)
    dim = 5
    info = quiver.feature.PartitionInfo(device, host, hosts, global2host)
    size = 10
    comm = quiver.comm.NcclComm(rank, ws, id, hosts, 1)
    feat = torch.ones(
        (size, dim), device=device, dtype=torch.float) * (1 + rank) * size
    for i in range(size):
        feat[i] += i
    print(f"{rank} hold {feat}")
    dist_feat = quiver.feature.DistFeature(feat, info, comm)
    ids = torch.randint(high=size,
                        size=(20, ),
                        dtype=torch.int64,
                        device=device)
    print(f"{rank} request global {ids}")
    host2ids, _ = dist_feat.info.dispatch(ids)
    print(f"{rank} request local {host2ids}")
    host2feats = dist_feat.comm.exchange(host2ids, feat)
    print(f"{rank} receive {host2feats}")


def child_feat_partition_pair(rank, ws, id, device, host, hosts, global2host):
    torch.cuda.set_device(device)
    dim = 400
    info = quiver.feature.PartitionInfo(device, host, hosts, global2host)
    nodes = 10000000
    comm = quiver.comm.NcclComm(rank, ws, id, hosts, 2)
    feat = torch.ones(
        (nodes, dim), device=device, dtype=torch.float) * (1 + rank)
    dist_feat = quiver.feature.DistFeature(feat, info, comm)
    size = 1000000
    ids = torch.randint(high=nodes,
                        size=(size, ),
                        dtype=torch.int64,
                        device=device)

    print('ready')
    host2ids, _ = dist_feat.info.dispatch(ids)
    for h, ids in enumerate(host2ids):
        print(f"{h} size {ids.size(0)}")
    host2feats = dist_feat.comm.exchange(host2ids, feat)
    num = 100
    print('test once')

    t0 = time.time()
    for i in range(num):
        print(i)
        beg = time.time()
        host2ids, _ = dist_feat.info.dispatch(ids)
        mid = time.time()
        host2feats = dist_feat.comm.exchange(host2ids, feat)
        end = time.time()
        print(f"dispatch {mid - beg}")
        print(f"exchange {end - mid}")
    t1 = time.time()
    if rank == 0:
        print(f"Latency {(t1 - t0) / num}")
        print(
            f"Throughput {num * size * dim * 4 / 1024 / 1024 / 1024 / (t1 - t0)}"
        )


def test_feat_partition():
    id = quiver.comm.getNcclId()
    ws = 2
    size = 10
    global2host = torch.randint(high=ws, size=(size, ), dtype=torch.int64)
    print(f"g2h {global2host}")
    procs = []
    for i in range(ws):
        proc = mp.Process(target=child_feat_partition,
                          args=(i, ws, id, i, i, ws, global2host))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()


def test_feat_partition_pair(rank):
    local_size = 2
    ws = 6
    store = dist.TCPStore(MASTER_ADDR, MASTER_PORT, 3,
                          MASTER_ADDR == LOCAL_ADDR)
    if rank == 0:
        id = quiver.comm.getNcclId()
        store.set("id", id)
    else:
        id = store.get("id")
    print(f"{rank} init store {id}")
    size = 10000000
    global2host = torch.randint(high=3, size=(size, ), dtype=torch.int64)
    procs = []
    for i in range(local_size):
        proc = mp.Process(target=child_feat_partition_pair,
                          args=(i + rank * local_size, ws, id, i, rank, 3,
                                global2host))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    # test_local()
    # test_dist_pair_bidirect(0)
    # test_nccl_allreduce_pair(0)
    test_feat_partition()
    # test_feat_partition_pair(0)
