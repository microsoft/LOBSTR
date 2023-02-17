"""
LOBSTR_SKFORMULA model and criterion classes.

Copyright (C) 2023 Microsoft Corporation

See detr/models/detr.py for additional information.
"""
from itertools import permutations
from collections import defaultdict
import sys

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

sys.path.append("../../lobstr")
from detr.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size)

from detr.models.backbone import build_backbone
from matcher import build_matcher
from detr.models.transformer import build_transformer


class LOBSTR_SKFORMULA(nn.Module):
    """ This is the LOBSTR module that performs graph recognition """
    def __init__(self, backbone, transformer, num_node_classes, num_hydrogens_classes,
                 num_formal_charge_classes, num_bond_classes, num_bond_direction_classes, num_queries,
                 aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_node_classes: number of node classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         LOBSTR can detect in a single image.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.node_class_embed = nn.Linear(hidden_dim, num_node_classes + 1)
        self.node_hydrogens_class_embed = nn.Linear(hidden_dim, num_hydrogens_classes + 1)
        self.node_formal_charge_class_embed = nn.Linear(hidden_dim, num_formal_charge_classes + 1)
        self.node_coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.node_adjacency_embed = nn.Linear(hidden_dim, num_queries * num_bond_classes)
        self.bond_direction_embed = nn.Linear(hidden_dim, num_queries * num_bond_direction_classes) 
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        node_outputs_class = self.node_class_embed(hs)
        node_hydrogens_outputs_class = self.node_hydrogens_class_embed(hs)
        node_formal_charge_outputs_class = self.node_formal_charge_class_embed(hs)
        node_outputs_class = self.node_class_embed(hs)
        node_outputs_coord = self.node_coord_embed(hs).sigmoid()
        node_outputs_adjacencies = self.node_adjacency_embed(hs)
        node_outputs_bond_directions = self.bond_direction_embed(hs)
        
        out = {'pred_node_logits': node_outputs_class[-1],
               'pred_hydrogens_logits': node_hydrogens_outputs_class[-1],
               'pred_formal_charge_logits': node_formal_charge_outputs_class[-1],
               'pred_coords': node_outputs_coord[-1],
               'pred_adjacencies': node_outputs_adjacencies[-1],
               'pred_bond_directions': node_outputs_bond_directions[-1]}
        
        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for LOBSTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_hydrogens_classes, num_formal_charge_classes,
                 matcher, weight_dict, eos_coef, losses, emphasized_weights={}, emphasized_hydrogens_weights={},
                 emphasized_formal_charge_weights={}):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_hydrogens_classes = num_hydrogens_classes
        self.num_formal_charge_classes = num_formal_charge_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        for class_num, weight in emphasized_weights.items():
            empty_weight[class_num] = weight
        self.register_buffer('empty_weight', empty_weight)
        empty_hydrogens_weight = torch.ones(self.num_hydrogens_classes + 1)
        empty_hydrogens_weight[-1] = self.eos_coef
        for class_num, weight in emphasized_hydrogens_weights.items():
            empty_hydrogens_weight[class_num] = weight
        self.register_buffer('empty_hydrogens_weight', empty_hydrogens_weight)
        empty_formal_charge_weight = torch.ones(self.num_formal_charge_classes + 1)
        empty_formal_charge_weight[-1] = self.eos_coef
        for class_num, weight in emphasized_formal_charge_weights.items():
            empty_formal_charge_weight[class_num] = weight
        self.register_buffer('empty_formal_charge_weight', empty_formal_charge_weight)

    def loss_labels(self, outputs, targets, indices, num_coords, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_coords]
        """
        
        losses = {}
        
        idx = self._get_src_permutation_idx(indices)
        
        assert 'pred_node_logits' in outputs
        src_logits = outputs['pred_node_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        
        assert 'pred_hydrogens_logits' in outputs
        src_logits = outputs['pred_hydrogens_logits']
        target_classes_o = torch.cat([t["hydrogens_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_hydrogens_classes-1,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce += F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_hydrogens_weight)
        
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['hydrogens_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        
        assert 'pred_formal_charge_logits' in outputs
        src_logits = outputs['pred_formal_charge_logits']
        target_classes_o = torch.cat([t["formal_charge_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_formal_charge_classes-1,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce += F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_formal_charge_weight)
        
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['formal_charge_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        
        losses['loss_ce'] = loss_ce

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_coords):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty coords
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_node_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_coords(self, outputs, targets, indices, num_coords):
        """Compute the losses related to the position, the L1 regression loss
           targets dicts must contain the key "coords" containing a tensor of dim [nb_target_coords, 2]
           The target coords are expected in format (center_x, center_y), normalized by the image size.
        """
        assert 'pred_coords' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_coords = outputs['pred_coords'][idx]
        target_coords = torch.cat([t['coords'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_coord = F.l1_loss(src_coords, target_coords, reduction='none')
        loss_coord = torch.pow(torch.pow(loss_coord[:, 0], 2) + torch.pow(loss_coord[:, 1], 2), 0.5)

        losses = {}
        losses['loss_coord'] = loss_coord.sum() / num_coords

        return losses
    
    # TODO
    # Need to permute in both directions
    def loss_adjacencies(self, outputs, targets, indices, num_coords):
        """Adjacency loss
        targets dicts must contain the key "adjacencies"
        """
        assert 'pred_adjacencies' in outputs
        assert 'pred_bond_directions' in outputs
        # Get the idx of the predictions that were matched to the target
        idx = self._get_src_permutation_idx(indices)
        
        batch_size = len(indices)
        
        loss_adj = 0.0
        for batch_num in range(batch_size):
            prediction_idx = indices[batch_num][0]
            target_idx = indices[batch_num][1]
        
            # Collect the matching predictions
            src_adjacencies = outputs['pred_adjacencies'][batch_num]
            src_adjacencies = src_adjacencies.reshape((src_adjacencies.shape[0],
                                                       src_adjacencies.shape[0], -1))
            src_adjacencies = src_adjacencies[np.ix_(prediction_idx, prediction_idx, range(src_adjacencies.shape[2]))]
            
            src_bond_directions = outputs['pred_bond_directions'][batch_num]
            src_bond_directions = src_bond_directions.reshape((src_bond_directions.shape[0],
                                                       src_bond_directions.shape[0], -1))
            src_bond_directions = src_bond_directions[np.ix_(prediction_idx, prediction_idx, range(src_bond_directions.shape[2]))]
            
            selected_adjacencies_targets = targets[batch_num]['adjacencies'][np.ix_(target_idx, target_idx)]
            selected_bond_directions_targets = targets[batch_num]['bond_directions'][np.ix_(target_idx, target_idx)]
            
            try:
                for query_num in range(src_adjacencies.shape[0]):
                    this_src = src_adjacencies[query_num, :, :].squeeze()
                    this_target = selected_adjacencies_targets[query_num, :].squeeze().long()
                    loss_adj += F.cross_entropy(this_src, this_target)
                    
                    this_src = src_bond_directions[query_num, :, :].squeeze()
                    this_target = selected_bond_directions_targets[query_num, :].squeeze().long()
                    loss_adj += F.cross_entropy(this_src, this_target)
            except:
                print(this_src.shape)
                print(this_target.shape)
                loss_adj = 0
                    
            #loss_bce += F.binary_cross_entropy(src_adjacencies, selected_targets.float())
        #losses_bce.append(loss_bce)
        losses = {'loss_adj': loss_adj / src_adjacencies.shape[0]}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_coords, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'coords': self.loss_coords,
            'adj': self.loss_adjacencies
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_coords, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target coords accross all nodes, for normalization purposes
        num_coords = sum(len(t["labels"]) for t in targets)
        num_coords = torch.as_tensor([num_coords], dtype=torch.float, device=next(iter(outputs.values())).device)
        #if is_dist_avail_and_initialized():
        #    torch.distributed.all_reduce(num_coords)
        num_coords = torch.clamp(num_coords / get_world_size(), min=1).item()
        
        for b in range(outputs['pred_node_logits'].shape[0]):
            new_outputs = {}
            for key in outputs.keys():
                new_outputs[key] = outputs[key][b, :, :].unsqueeze(0)
            #new_targets = {}
            #for key in targets.keys():
            #    new_targets[key] = targets[key][0, :, :]
            #print(len(targets))
            new_targets = [targets[b]]
            new_indices = [indices[b]]
            
            target_coords = torch.cat([t['coords'][i] for t, (_, i) in zip(new_targets, new_indices)], dim=0)
            target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(new_targets, new_indices)])
            target_hydrogens = torch.cat([t["hydrogens_labels"][J] for t, (_, J) in zip(new_targets, new_indices)])
            target_charges = torch.cat([t["formal_charge_labels"][J] for t, (_, J) in zip(new_targets, new_indices)])
            target_coords = target_coords.cpu().tolist()
            target_classes = target_classes.cpu().tolist()
            group_by_location = defaultdict(set)
            for idx1, (coord, class_label, hydrogen_label, charge_label) in enumerate(zip(target_coords,
                                                                                          target_classes,
                                                                                          target_hydrogens,
                                                                                          target_charges)):
                group_by_location[tuple(coord + [class_label, hydrogen_label, charge_label])].add(idx1)

            for c, s in group_by_location.items():
                if len(s) > 1:
            
                    lowest_loss = 1000000000

                    v = new_indices[0][1][list(s)].tolist()
                    perm = permutations(s)

                    for i in list(perm): 
                        for o, n in zip(i, v):
                            new_indices[0][1][o] = n

                        # Compute all the requested losses
                        losses = {}
                        for loss in self.losses:
                            losses.update(self.get_loss(loss, new_outputs, new_targets, new_indices, num_coords))
                        loss_total = sum(losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict)
                        print(loss_total)
                        if loss_total.item() < lowest_loss:
                            lowest_loss = loss_total.item()
                            best_indices = new_indices[0][1].detach().clone()
                            indices[b] = (indices[b][0], best_indices)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_coords))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_coords, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_coord = outputs['pred_node_logits'], outputs['pred_coords']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h], dim=1)
        coords = coords * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'coords': b} for s, l, b in zip(scores, labels, coords)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = LOBSTR_SKFORMULA(backbone, transformer,
                             args.num_node_classes, args.num_hydrogens_classes, 
                             args.num_formal_charge_classes, args.num_bond_classes, 
                             args.num_bond_direction_classes, args.num_queries)

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_coord': args.coord_loss_coef, 'loss_adj': args.adj_loss_coef}
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    losses = ['labels', 'coords', 'cardinality', 'adj']
    criterion = SetCriterion(args.num_node_classes, args.num_hydrogens_classes, args.num_formal_charge_classes,
                             matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, emphasized_weights=args.emphasized_weights,
                             emphasized_hydrogens_weights=args.emphasized_hydrogens_weights,
                             emphasized_formal_charge_weights=args.emphasized_formal_charge_weights)
    criterion.to(device)
    postprocessors = {'coord': PostProcess()}

    return model, criterion, postprocessors
