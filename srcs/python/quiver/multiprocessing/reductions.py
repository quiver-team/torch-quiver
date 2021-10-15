from multiprocessing.reduction import ForkingPickler
import quiver

def rebuild_feature(ipc_handle):

    feature = quiver.Feature.lazy_from_ipc_handle(ipc_handle)
    return feature

def reduce_feature(feature):
    
    ipc_handle = feature.share_ipc()
    return (rebuild_feature, (ipc_handle, ))


def rebuild_pyg_sampler(cls, ipc_handle):
    sampler = cls.lazy_from_ipc_handle(ipc_handle)
    return sampler
    

def reduce_pyg_sampler(sampler):
    ipc_handle = sampler.share_ipc()
    return (rebuild_pyg_sampler, (type(sampler), ipc_handle, ))
  



def init_reductions():
    ForkingPickler.register(quiver.Feature, reduce_feature)
    ForkingPickler.register(quiver.pyg.GraphSageSampler, reduce_pyg_sampler)