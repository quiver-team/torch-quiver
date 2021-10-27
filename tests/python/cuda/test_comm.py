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
    for i in range(num):
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
    for i in range(num):
        if rank % 2 == 0:
            comm.send(large, rank + 1)
        else:
            comm.recv(large, rank - 1)
        torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(num):
        if rank == 0:
            comm.send(large, rank + 1)
        else:
            comm.recv(large, rank - 1)
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


def child_torch_allreduce_proc(rank):
    torch.cuda.set_device(rank)
    print(f"{rank} ready")
    if rank == 0:
        a = torch.zeros(10, device=0)
    else:
        a = torch.ones(10, device=1)
    dist.all_reduce(a)
    print(f"{rank} tensor {a}")
    num = 10
    size = 1024 * 1024 * 1024
    large = torch.zeros(size, device=rank)
    for i in range(5):
        dist.all_reduce(large)
        torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(num):
        dist.all_reduce(large)
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


def test_torch_allreduce(rank):
    ws = 2
    dist.init_process_group('nccl', rank=rank, world_size=2)
    child_torch_allreduce_proc(rank)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    # test_local()
    test_dist_pair(1)
    # test_torch_allreduce(0)