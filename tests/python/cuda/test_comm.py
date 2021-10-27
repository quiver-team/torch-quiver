import torch
import torch.multiprocessing as mp
import torch_quiver as torch_qv
import quiver

def child_proc(rank, ws, id):
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

if __name__ == "__main__":
    mp.set_start_method('spawn')
    id = quiver.comm.getNcclId()
    ws = 2
    procs = []
    for i in range(ws):
        proc = mp.Process(target=child_proc, args=(i, ws, id))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()