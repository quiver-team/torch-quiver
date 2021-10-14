
def find_cliques(adj_mat, clique_res, potential_clique=[], remaining_nodes=[], skip_nodes=[]):

    if len(remaining_nodes) == 0 and len(skip_nodes) == 0:
        clique_res.append(potential_clique)
        return 1

    found_cliques = 0
    for node in remaining_nodes:

        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if adj_mat[node][n] == 1]
        new_skip_list = [n for n in skip_nodes if adj_mat[node][n] == 1]
     
        found_cliques += find_cliques(adj_mat, clique_res, new_potential_clique, new_remaining_nodes, new_skip_list)

        # We're done considering this node.  If there was a way to form a clique with it, we
        # already discovered its maximal clique in the recursive call above.  So, go ahead
        # and remove it from the list of remaining nodes and add it to the skip list.
        remaining_nodes.remove(node)
        skip_nodes.append(node)
    return found_cliques


all_nodes = [0, 1, 2, 3]
adj_mat = [
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 0],
]
clique_res = []
find_cliques(adj_mat, clique_res, remaining_nodes=all_nodes)
print(clique_res)
"""
class Node(object):

    def __init__(self, name):
        self.name = name
        self.neighbors = []

    def __repr__(self):
        return self.name


A = Node('A')
B = Node('B')
C = Node('C')
D = Node('D')
E = Node('E')
F = Node('F')

A.neighbors = [B, C, E]
B.neighbors = [A, C, D, F]
C.neighbors = [A, B, D, F]
D.neighbors = [C, B, E, F]
E.neighbors = [A, D]
F.neighbors = [B, C, D]

all_nodes = [A, B, C, D, E, F]


def find_cliques(potential_clique=[], remaining_nodes=[], skip_nodes=[], depth=0):

    if len(remaining_nodes) == 0 and len(skip_nodes) == 0:
        print('This is a clique:', potential_clique)
        return 1

    found_cliques = 0
    for node in remaining_nodes:

        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in node.neighbors]
        new_skip_list = [n for n in skip_nodes if n in node.neighbors]
        print(new_remaining_nodes)
        print(new_skip_list)
        exit()
        found_cliques += find_cliques(new_potential_clique, new_remaining_nodes, new_skip_list, depth + 1)

        # We're done considering this node.  If there was a way to form a clique with it, we
        # already discovered its maximal clique in the recursive call above.  So, go ahead
        # and remove it from the list of remaining nodes and add it to the skip list.
        remaining_nodes.remove(node)
        skip_nodes.append(node)
    return found_cliques

total_cliques = find_cliques(remaining_nodes=all_nodes)
print('Total cliques found:', total_cliques)
"""