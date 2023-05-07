# Documentation for `torch-quiver`

::: quiver.Feature
    handler: python
    selection:
      members:
        - from_cpu_tensor
        - from_mmap
        - set_local_order
        - size
        - share_ipc
        - new_from_ipc_handle
        - dim
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.DistFeature
    handler: python
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.NcclComm
    handler: python
    selection:
      members:
        - exchange
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.PartitionInfo
    handler: python
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.pyg.GraphSageSampler
    handler: python
    selection:
        members:
            - sample
            - share_ipc
            - lazy_from_ipc_handle
    rendering:
        show_source: true
        show_root_heading: true

::: quiver.CSRTopo
    handler: python
    selection:
      members:
        - indptr
        - indices
        - feature_order
        - degree
        - node_count
        - edge_count
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.init_p2p
    handler: python
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.p2pCliqueTopo
    handler: python
    selection:
      members:
        - info
        - p2p_clique
        - get_clique_id
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.partition.partition_with_replication
    handler: python
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.partition.partition_without_replication
    handler: python
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.partition.partition_free
    handler: python
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.AutoBatch
    handler: python
    selection:
      members:
        - get_batched_queue
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.ServingSampler
    handler: python
    selection:
      members:
        - lazy_init
        - get_sampled_queue
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.ServerInference
    handler: python
    selection:
      members:
        - lazy_init
        - get_inferenced_queue
    rendering:
      show_source: true
      show_root_heading: true

::: quiver.generate_neighbour_num
    handler: python
    rendering:
      show_source: true
      show_root_heading: true
