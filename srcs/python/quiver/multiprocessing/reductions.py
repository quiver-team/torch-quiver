from multiprocessing.reduction import ForkingPickler
import quiver

def rebuild_feature(ipc_handle):

    feature = quiver.Feature.lazy_from_ipc_handle(ipc_handle)
    return feature

def reduce_feature(feature):
    
    ipc_handle = feature.share_ipc()
    return (rebuild_feature, (ipc_handle, ))

def reduce_sampler():
    pass

def rebuild_sampler():
    pass    



def init_reductions():
    ForkingPickler.register(quiver.Feature, reduce_feature)