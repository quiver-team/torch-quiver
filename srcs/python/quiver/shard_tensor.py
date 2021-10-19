import torch_quiver as torch_qv
import torch

from .utils import Topo

class Offset:
    def __init__(self, start, end):
        self.start_ = start
        self.end_ = end

    @property
    def start(self):
        return self.start_

    @property
    def end(self):
        return self.end_


class DeviceCollectionJob:
    def __init__(self, part_orders, request_nodes):
        self.part_orders_ = part_orders
        self.request_nodes_ = request_nodes

    @property
    def part_orders(self):
        return self.part_orders_

    @property
    def request_nodes(self):
        return self.request_nodes_


class ShardTensorConfig:
    """
    """
    def __init__(self, device_memory_budget):
        self.tensor_offset_device = {}

        self.device_memory_budget = device_memory_budget
        for device in device_memory_budget:
            if isinstance(self.device_memory_budget[device], int):
                continue
            elif isinstance(self.device_memory_budget[device], float):
                self.device_memory_budget[device] = int(
                    self.device_memory_budget[device])

            elif isinstance(self.device_memory_budget[device], str):
                if self.device_memory_budget[device].upper().endswith(
                        "M") or self.device_memory_budget[device].upper(
                        ).endswith("MB"):
                    end = -1 if self.device_memory_budget[device].upper(
                    ).endswith("M") else -2
                    self.device_memory_budget[device] = int(
                        float(self.device_memory_budget[device][:end]) * 1024 *
                        1024)

                elif self.device_memory_budget[device].upper().endswith(
                        "G") or self.device_memory_budget[device].upper(
                        ).endswith("GB"):
                    end = -1 if self.device_memory_budget[device].upper(
                    ).endswith("G") else -2
                    self.device_memory_budget[device] = int(
                        float(self.device_memory_budget[device][:end]) * 1024 *
                        1024 * 1024)
            else:
                raise Exception("memory budget input is not valid")
            print(
                f"LOG >>> Memory Budge On {device} is {self.device_memory_budget[device] // 1024 // 1024}MB"
            )

    @property
    def device_list(self):
       
        return list(self.device_memory_budget.keys())


class ShardTensor:
    """[summary]
    """
    def __init__(self, current_device: int,
                 shard_tensor_config: ShardTensorConfig):
        self.shard_tensor = torch_qv.ShardTensor(current_device)
        self.current_device = current_device
        self.shard_tensor_config = shard_tensor_config or ShardTensorConfig({})
        self.topo = None
        self.current_clique = None

        # cpu part
        self.cpu_tensor = None

    def init_topo(self):
        if self.current_clique is not None:
            return

        device_list = set(self.shard_tensor_config.device_list)
        device_list.add(self.current_device)
        device_list = list(device_list)
        self.topo = Topo(device_list)
        self.current_clique = self.topo.get_clique_id(self.current_device)

    def append(self, cpu_tensor, device):

        if device == -1:
            if self.cpu_tensor is not None:
                raise Exception("cpu tensor has been already appended")
            self.cpu_tensor = cpu_tensor
            self.shard_tensor.append(cpu_tensor, -1)
            return
        if self.shard_tensor_config.device_memory_budget.get(device,
                                                             None) is None:
            self.shard_tensor_config.tensor_offset_device[device] = Offset(
                self.shard_tensor.size(0),
                self.shard_tensor.size(0) + cpu_tensor.shape[0])
            self.shard_tensor_config.device_memory_budget[
                device] = cpu_tensor.numel() * 4
            print(
                f"LOG >>> Memory Budge On {device} is {self.shard_tensor_config.device_memory_budget[device] // 1024 // 1024} MB"
            )
            self.shard_tensor.append(cpu_tensor, device)
        else:
            raise Exception(f"{device} tensor has been already appended")

    def partition(self, tensor, memory_budget):
        """
        Args:
            tensor: pytorch cpu tensor
            memory_budget: memory size in bytes
            
        """
        # 暂时先假设为float tensor
        element_size = tensor.shape[1] * 4
        return memory_budget // element_size

    def from_cpu_tensor(self, tensor):
        cur_pos = 0
        size = 0
        # We Assume Only 2 Numa Node
        for device_id, memory_budget in self.shard_tensor_config.device_memory_budget.items(
        ):
            if cur_pos > tensor.shape[0]:
                break

            size = self.partition(tensor, memory_budget)
            size = min(size, tensor.shape[0] - cur_pos)
            self.shard_tensor.append(tensor[cur_pos:cur_pos + size], device_id)
            device_offset = Offset(cur_pos, cur_pos + size)
            self.shard_tensor_config.tensor_offset_device[
                device_id] = device_offset

            cur_pos += size
            print(
                f"LOG >>> Assign {int(100 * size * 1.0 / tensor.shape[0])}% data to {device_id}"
            )

        if cur_pos < tensor.shape[0]:
            # allocate the rest of data on CPU
            self.cpu_tensor = tensor[cur_pos: ]
            self.shard_tensor.append(self.cpu_tensor, -1)
            print(
                f"LOG >>> Assign {100 - int(100 * cur_pos * 1.0 / tensor.shape[0])}% data to CPU"
            )
            del tensor

    def collect_device(self, input_orders, nodes, inter_device, wait_results):

        request_nodes_mask = (nodes >= self.shard_tensor_config.tensor_offset_device[inter_device].start) & (
                                    nodes < self.shard_tensor_config.tensor_offset_device[inter_device].end)
        request_nodes = torch.masked_select(nodes, request_nodes_mask)
        part_orders = torch.masked_select(input_orders, request_nodes_mask)
        request_nodes = request_nodes.to(inter_device)

        with torch.cuda.device(inter_device):
            result = self.shard_tensor[request_nodes]
        result = result.to(self.current_device)
        wait_results.append((part_orders, result))

    def __getitem__(self, nodes):

        self.init_topo()
        nodes = nodes.to(self.current_device)

        feature = self.shard_tensor[nodes]

        input_orders = torch.arange(nodes.size(0), dtype=torch.long, device=self.current_device)

        # call inter request, we unfold for loop
        inter_clique_devices = self.topo.p2pClique2Device.get(1 - self.current_clique, [])

        wait_results = []

        for inter_device in inter_clique_devices:
            if self.shard_tensor_config.tensor_offset_device.get(inter_device, None) is not None:
                self.collect_device(input_orders, nodes, inter_device, wait_results)
                
        for result in wait_results:
            feature[result[0]] = result[1]

        return feature

    @property
    def shape(self):
        return self.shard_tensor.shape()

    @property
    def device(self):
        return self.current_device

    def share_ipc(self):
        items = self.shard_tensor.share_ipc()
        gpu_part_ipc_list = [item.share_ipc() for item in items]

        return gpu_part_ipc_list, self.cpu_tensor, self.shard_tensor_config

    def from_ipc_handle(self, gpu_ipc_list, cpu_tensor):
        for gpu_ipc in gpu_ipc_list:
            gpu_item = torch_qv.ShardTensorItem()
            gpu_item.from_ipc(gpu_ipc)
            self.shard_tensor.append(gpu_item)
        if cpu_tensor is not None:
            self.cpu_tensor = cpu_tensor
            self.shard_tensor.append(cpu_tensor, -1)

    @classmethod
    def new_from_share_ipc(cls, ipc_handles, current_device):
        gpu_part_ipc_list, cpu_tensor, shard_tensor_config = ipc_handles
        shard_tensor = cls(current_device, shard_tensor_config)
        shard_tensor.from_ipc_handle(gpu_part_ipc_list, cpu_tensor)
        return shard_tensor
    
    def size(self, dim):
        return self.shard_tensor.size(dim)

