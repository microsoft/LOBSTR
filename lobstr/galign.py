import multiprocessing
import random
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.optimize import linear_sum_assignment

def dist(nodes1, nodes2, adjacencies1, adjacencies2, cost_class=1, cost_adj=0.25):
    score1 = np.sum((nodes1 - nodes2)**2) / (np.sum(nodes1) + np.sum(nodes2))
    score2 = np.sum((adjacencies1 - adjacencies2)**2) / (np.sum(adjacencies1) + np.sum(adjacencies2))
    
    return cost_class * score1 + cost_adj * score2

reward_function = lambda x, y: 1 / (1 + abs(x - y))

def align_1d(sequence1, sequence2, reward_function, return_alignment=False):
    '''
    Dynamic programming sequence alignment between two sequences
    Traceback convention: -1 = up, 1 = left, 0 = diag up-left
    '''
    sequence1_length = len(sequence1)
    sequence2_length = len(sequence2)
    
    scores = np.zeros((sequence1_length + 1, sequence2_length + 1))
    pointers = np.zeros((sequence1_length + 1, sequence2_length + 1))
    
    # Initialize first column
    for row_idx in range(1, sequence1_length + 1):
        pointers[row_idx, 0] = -1
        
    # Initialize first row
    for col_idx in range(1, sequence2_length + 1):
        pointers[0, col_idx] = 1  
        
    for row_idx in range(1, sequence1_length + 1):
        for col_idx in range(1, sequence2_length + 1):
            reward = reward_function(sequence1[row_idx-1], sequence2[col_idx-1])
            diag_score = scores[row_idx - 1, col_idx - 1] + reward
            same_row_score = scores[row_idx, col_idx - 1]
            same_col_score = scores[row_idx - 1, col_idx]
               
            max_score = max(diag_score, same_col_score, same_row_score)
            scores[row_idx, col_idx] = max_score
            if diag_score == max_score:
                pointers[row_idx, col_idx] = 0
            elif same_col_score == max_score:
                pointers[row_idx, col_idx] = -1
            else:
                pointers[row_idx, col_idx] = 1
    
    score = scores[sequence1_length, sequence2_length]
    score = 2 * score / (sequence1_length + sequence2_length)
    
    if not return_alignment:
        return score
    
    # Backtrace
    cur_row = sequence1_length
    cur_col = sequence2_length
    aligned_sequence1_indices = []
    aligned_sequence2_indices = []
    while not (cur_row == 0 and cur_col == 0):
        if pointers[cur_row, cur_col] == -1:
            cur_row -= 1
        elif pointers[cur_row, cur_col] == 1:
            cur_col -= 1
        else:
            cur_row -= 1
            cur_col -= 1
            aligned_sequence1_indices.append(cur_col)
            aligned_sequence2_indices.append(cur_row)
            
    aligned_sequence1_indices = aligned_sequence1_indices[::-1]
    aligned_sequence2_indices = aligned_sequence2_indices[::-1]
    
    return aligned_sequence1_indices, aligned_sequence2_indices, score

# Iteratively swaps pairs of nodes to improve the alignment
def iterative_pair_swap_graph_alignment(nodes1, nodes2, adjacencies1, adjacencies2, use_distances=False):
    if use_distances:
        graph1 = csr_matrix(adjacencies1)
        dist_matrix1 = floyd_warshall(csgraph=graph1, directed=False, return_predecessors=False)
        adjacencies1 = dist_matrix1
        graph2 = csr_matrix(adjacencies2)
        dist_matrix2 = floyd_warshall(csgraph=graph2, directed=False, return_predecessors=False)
        adjacencies2 = dist_matrix2
    
    size1 = nodes1.shape[0]
    size2 = nodes2.shape[0]
    current_dist = dist(nodes1, nodes2, adjacencies1, adjacencies2)
    previous_dist = current_dist
    nodes2_alignment = np.array(range(size2))
    best_indices = range(size2)

    while(True):
        for idx1 in range(size2-1):
            for idx2 in range(idx1+1, size2):
                indices = nodes2_alignment.copy()
                temp = indices[idx1]
                indices[idx1] = indices[idx2]
                indices[idx2] = temp
                score = dist(nodes1, nodes2[indices], adjacencies1, adjacencies2[np.ix_(indices, indices)])

                if score < current_dist:
                    current_dist = score
                    best_indices = indices

        if current_dist < previous_dist:
            print(current_dist)
            nodes2_alignment = best_indices
            previous_dist = current_dist
            
            if current_dist == 0:
                break
        else:
            break
            
    return nodes2_alignment, previous_dist

def match_nodes(distances1, distances2, node1_num, node2_num):
    sorted_distances1 = sorted(distances1[:, node1_num])
    sorted_distances2 = sorted(distances2[:, node2_num])
    d = align_1d(sorted_distances1, sorted_distances2, reward_function)
    
    return d

# Hungarian-based all-pairs distance matching
def hungarian_distance_graph_align(nodes1, nodes2, adjacencies1, adjacencies2):
    graph1 = csr_matrix(adjacencies1)
    distances1 = floyd_warshall(csgraph=graph1, directed=False, return_predecessors=False)
    graph2 = csr_matrix(adjacencies2)
    distances2 = floyd_warshall(csgraph=graph2, directed=False, return_predecessors=False)  
    
    size1 = nodes1.shape[0]
    size2 = nodes2.shape[0]

    matching_matrix = np.zeros((size1, size2))
    
    values = []
    for node1_num in range(size1):
        for node2_num in range(size2):
            values.append((distances1, distances2, node1_num, node2_num))

    with multiprocessing.Pool() as pool:
        res = pool.starmap(match_nodes, values)

    for d, v in zip(res, values):
        matching_matrix[v[2], v[3]] = d
            
    row_ind, nodes2_alignment = linear_sum_assignment(matching_matrix, maximize=True)
    
    score = dist(nodes1, nodes2[nodes2_alignment], distances1, distances2[np.ix_(nodes2_alignment, nodes2_alignment)])
            
    #return row_ind, col_ind #nodes2_alignment
    return nodes2_alignment, score

def hybrid_graph_alignment(nodes1, nodes2, adjacencies1, adjacencies2):
    nodes2_alignment, distance = hungarian_distance_graph_align(nodes1, nodes2, adjacencies1, adjacencies2)
    
    if distance > 0:
        print(distance)
        nodes2 = nodes2[nodes2_alignment]
        adjacencies2 = adjacencies2[np.ix_(nodes2_alignment, nodes2_alignment)]
        nodes2_alignment2, distance = iterative_pair_swap_graph_alignment(nodes1, nodes2, adjacencies1, adjacencies2, use_distances=True)
        nodes2_alignment = nodes2_alignment[nodes2_alignment2]
        
    return nodes2_alignment, distance

nodes1 = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1],
          [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1],
          [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1],
          [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1],
          [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1],
          [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1],
          [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]]
nodes1 = np.array(nodes1)
size1 = nodes1.shape[0]
adjacencies1 = np.random.choice([0, 1], size=(size1, size1), p=[0.8, 0.2])

for idx1 in range(size1 - 1):
    for idx2 in range(idx1+1, size1):
        adjacencies1[idx2, idx1] = adjacencies1[idx1, idx2]
print(adjacencies1)

indices = np.random.permutation(nodes1.shape[0])
print(indices)
nodes2 = nodes1[indices]

adjacencies2 = adjacencies1[np.ix_(indices, indices)]

for x in range(30):
    idx1 = random.randint(0, size1-1)
    idx2 = random.randint(0, size1-1)
    value = 1 - adjacencies2[idx1, idx2]
    adjacencies2[idx1, idx2] = value
    adjacencies2[idx2, idx1] = value
inverse_indices = np.argsort(indices)
print(inverse_indices)

st = time.time()
nodes2_alignment, distance = hybrid_graph_alignment(nodes1, nodes2, adjacencies1, adjacencies2)
et = time.time()
print("Distance: {}".format(distance))
print("Time: {}".format(et - st))

nodes1_alignment = np.argsort(nodes2_alignment)

print(inverse_indices)
print(nodes2_alignment)
print(inverse_indices == nodes2_alignment)
print(nodes1 == nodes2[nodes2_alignment])
