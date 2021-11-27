import torch
import numpy as np 
from scipy.sparse import csr_matrix

def load_liveJ():
    data_file = open("./com-lj.ungraph.txt")
    for _ in range(4):
        next(data_file)
    
    
    edge_index = np.zeros((2, 34681189 * 2), dtype=np.int64)
    edge_count = 0
    while True:
        
        try:
            line = next(data_file)
        except StopIteration:
            break
        data = line.split()
        src_id = int(data[0])
        dst_id = int(data[-1])
        
        edge_index[0][edge_count] = src_id
        edge_index[1][edge_count] = dst_id
        edge_count += 1

        edge_index[0][edge_count] = dst_id
        edge_index[1][edge_count] = src_id
        edge_count += 1

        
    
    
    data = np.zeros(34681189 * 2)

  

    csr_mat = csr_matrix((data, (edge_index[0], edge_index[1])))
    indptr = csr_mat.indptr
    indices = csr_mat.indices

    indptr = torch.from_numpy(indptr).type(torch.long)
    indices = torch.from_numpy(indices).type(torch.long)
    torch.save(indptr, "com-lj_indptr.pt")
    torch.save(indices, "com-lj_indices.pt")
        

    


        


load_liveJ()
    

            


