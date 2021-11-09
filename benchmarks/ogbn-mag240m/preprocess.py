from ogb.lsc import MAG240MDataset
from scipy.sparse import csr
import torch
import quiver


def get_nonzero():
    dataset = MAG240MDataset("/data/mag")

    train_idx = torch.from_numpy(dataset.get_idx_split('train'))

    path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
    adj_t = torch.load(path)
    indptr, indices, _ = adj_t.csr()
    del adj_t

    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 15],
                                                 0,
                                                 mode="UVA")

    prob = quiver_sampler.sample_prob(train_idx, indptr.size(0) - 1)
    nz = torch.nonzero(prob).to('cpu')
    print("nonzero")
    return nz

def preprocess():
    nz = get_nonzero()
    dataset = MAG240MDataset("/data/mag")
    x = dataset.paper_feat
    del dataset
    b = torch.from_numpy(x[nz])
    del x
    print(b.size())
    # time.sleep(30)
    # b = b.to(dtype=torch.float32)

preprocess()