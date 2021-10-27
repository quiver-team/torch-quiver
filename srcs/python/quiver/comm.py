import torch_quiver as torch_qv


class NcclComm:
    def __init__(self, rank, ws, id):
        self.comm = torch_qv.NcclComm(rank, ws, id)

    @property
    def rank(self):
        return self.comm.rank()

    @property
    def size(self):
        return self.comm.size()

    @property
    def device(self):
        return self.comm.device()

    def send(self, tensor, dst):
        return self.comm.send(tensor, dst)

    def recv(self, tensor, src):
        return self.comm.recv(tensor, src)

    def allreduce(self, tensor):
        return self.comm.allreduce(tensor)


def getNcclId():
    return torch_qv.create_nccl_id()