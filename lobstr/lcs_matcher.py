"""
LCS matcher
"""
import torch
from torch import nn
import numpy as np
import time
import multiprocessing

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

def align_1d(cost_matrix):
    '''
    Dynamic programming sequence alignment between two sequences
    Traceback convention: -1 = up, 1 = left, 0 = diag up-left

    i = prediction
    j = target
    '''
    sequence1_length, sequence2_length = cost_matrix.shape

    if sequence1_length == 0 or sequence2_length == 0:
        return []

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
            reward = 10000 - cost_matrix[row_idx-1, col_idx-1]
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
    #score = 2 * score / (sequence1_length + sequence2_length)

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
            aligned_sequence1_indices.append(cur_row)
            aligned_sequence2_indices.append(cur_col)

    aligned_sequence1_indices = aligned_sequence1_indices[::-1]
    aligned_sequence2_indices = aligned_sequence2_indices[::-1]

    return aligned_sequence1_indices, aligned_sequence2_indices

class LCSMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        
        #start_time = time.time()
        #indices = [self.align_1d(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        #end_time = time.time()
        #print(indices)
        #print("{} seconds to match.".format(end_time - start_time))
        
        #start_time = time.time()
        costs = [c[i] for i, c in enumerate(C.split(sizes, -1))]
        with multiprocessing.Pool(processes=bs) as pool:
            indices = pool.map(align_1d, costs)
        end_time = time.time()
        #print(indices)
        #print("{} seconds to match.".format(end_time - start_time))
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
    
    def align_1d(self, cost_matrix):
        '''
        Dynamic programming sequence alignment between two sequences
        Traceback convention: -1 = up, 1 = left, 0 = diag up-left

        i = prediction
        j = target
        '''
        sequence1_length, sequence2_length = cost_matrix.shape

        if sequence1_length == 0 or sequence2_length == 0:
            return []

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
                reward = 10000 - cost_matrix[row_idx-1, col_idx-1]
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
        #score = 2 * score / (sequence1_length + sequence2_length)

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
                aligned_sequence1_indices.append(cur_row)
                aligned_sequence2_indices.append(cur_col)

        aligned_sequence1_indices = aligned_sequence1_indices[::-1]
        aligned_sequence2_indices = aligned_sequence2_indices[::-1]

        return aligned_sequence1_indices, aligned_sequence2_indices


def build_lcs_matcher(args):
    return LCSMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
