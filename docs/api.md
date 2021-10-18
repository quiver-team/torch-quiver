# Documentation for `quiver.Feature`

::: quiver.Feature
    handler: python
    selection:
      members:
        - from_cpu_tensor
        - size
        - share_ipc
        - new_from_ipc_handle
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



