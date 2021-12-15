import torch
import torch_quiver as torch_qv


class HostRankTable:
    def __init__(self, hosts, rank_per_host):
        self.hosts = hosts
        self.rank_per_host = rank_per_host
        self.host2ranks = dict()
        self.rank2host = []
        cnt = 0
        for i in range(hosts):
            self.host2ranks[i] = list(range(cnt, cnt + rank_per_host))
            cnt += rank_per_host
            self.rank2host.extend([i] * rank_per_host)

    def ranks(self, host):
        return self.host2ranks[host]

    def host(self, rank):
        return self.rank2host[rank]

    def remote_peer(self, rank, host):
        remote_ranks = self.ranks(host)
        return remote_ranks[rank % self.rank_per_host]

    def remote_peers(self, rank, hosts):
        return [(rank, self.remote_peer(rank, host)) for host in hosts]

    def get_comm_mat(self, flat_allreduce):
        flat_allreduce = flat_allreduce.to('cpu')
        comm_mat = []
        size = self.hosts * self.rank_per_host
        for i in range(size):
            row = []
            for j in range(size):
                row.append(int(flat_allreduce[i * size + j]))
            comm_mat.append(row)
        return comm_mat


def schedule(comm_mat, table):
    steps = []
    cont = True
    host = table.hosts
    traversed_pair = set()
    while cont:
        step = []
        traversed_host = set()
        for src in range(host):
            if src in traversed_host:
                continue
            src_ranks = table.ranks(src)
            finished = False
            for dst in range(host):
                if dst in traversed_host:
                    continue
                if (src, dst) in traversed_pair:
                    continue
                traversed_pair.add((src, dst))
                for src_rank in src_ranks:
                    dst_rank = table.remote_peer(src_rank, dst)
                    if comm_mat[src_rank][dst_rank] <= 0:
                        continue
                    step.append((src_rank, dst_rank))
                    finished = True
                if finished:
                    traversed_host.add(src)
                    traversed_host.add(dst)
                    break
        if len(step) == 0:
            cont = False
        else:
            steps.append(step)
    return steps


class NcclComm:
    """NCCL Communicator is used to exchange data between worker processes. Essentially
    it is a wrapper of the communicator in the NCCL library. It supports both p2p
    communication (e.g. send/recv) and collective communication (e.g. allreduce).

    It is initialized by NCCL ID, which is provided by getNcclId. After this ID is synced
    across all workers in the cluster and every worker knows the cluster topology, then
    an NCCL communicator can be constructed with them.
    
    ```python
    >>> id = getNCCLId() # only one worker needs to get this and sync it to all peers
    >>> comm = NcclComm(rank, ws, id...)
    ```

    Args:
        rank (int): global rank of the worker
        ws (int): global world size
        id (bytes): NCCL unique ID
        hosts (int): the number of hosts in the cluster
        rank_per_host (int): the number of workers per host
        
    """
    def __init__(self, rank, ws, id, hosts=None, rank_per_host=None):
        self.comm = torch_qv.NcclComm(rank, ws, id)
        if hosts is not None:
            self.table = HostRankTable(hosts, rank_per_host)
            self.host = self.table.host(rank)

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
        self.comm.send(tensor, dst)

    def recv(self, tensor, src):
        self.comm.recv(tensor, src)

    def allreduce(self, tensor):
        self.comm.allreduce(tensor)

    def exchange(self, host2ids, feature):
        """Exchange features to remote peers. Each worker will send requests which contain
        remote feature IDs. After recerving requests, each worker will collect features
        and send back feature tensor responses to the peers who request features.

        Args:
            host2ids ([torch.Tensor]): list of remote IDs in requests
            feature (quiver.Feature): local feature object

        Returns:
            [torch.Tensor]: list of remote feature tensors in responses

        """
        remote_sizes = torch.zeros(self.size * self.size, dtype=torch.int64)
        for host in range(self.table.hosts):
            ids = host2ids[host]
            remote_peer = self.table.remote_peer(self.rank, host)
            if ids is not None and remote_peer != self.rank:
                remote_sizes[self.rank * self.size + remote_peer] = ids.size(0)
        remote_sizes = remote_sizes.to(self.device)
        self.allreduce(remote_sizes)
        comm_mat = self.table.get_comm_mat(remote_sizes)
        steps = schedule(comm_mat, self.table)
        torch.cuda.current_stream().synchronize()
        req_ids = [None] * self.size
        res_feats = [None] * self.size
        for step in steps:
            for src, dst in step:
                if src == self.rank:
                    ids = host2ids[self.table.host(dst)]
                    self.send(ids, dst)
                if dst == self.rank:
                    ids = torch.zeros(comm_mat[src][dst],
                                      dtype=torch.int64,
                                      device=self.device)
                    self.recv(ids, src)
                    req_ids[src] = ids
        torch.cuda.current_stream().synchronize()
        for i in range(len(req_ids)):
            ids = req_ids[i]
            if ids is not None:
                res_feats[i] = feature[ids]
        host2feats = [None] * self.table.hosts
        for step in steps:
            for src, dst in step:
                if dst == self.rank:
                    feats = res_feats[src]
                    self.send(feats, src)
                if src == self.rank:
                    feats = torch.zeros(comm_mat[src][dst],
                                        feature.size(1),
                                        device=self.device)
                    self.recv(feats, dst)
                    host2feats[self.table.host(dst)] = feats
        torch.cuda.current_stream().synchronize()
        return host2feats


def getNcclId():
    return torch_qv.create_nccl_id()
