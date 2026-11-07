"""
Pruning utilities used in phases 3-5: diagonal / gate Fisher, magnitude and OBD scores,
unstructured, N:M and structured (head / neuron) masks. Pruning = multiplying weights
by a 0/1 mask; there is no retraining.
"""

import torch
import torch.nn.functional as F

from gpt2_model import get_prunable_linear_layers


def save_original_weights(model):
    saved = {}
    for name, layer in get_prunable_linear_layers(model):
        saved[name + ".weight"] = layer.weight.detach().cpu().clone()
        saved[name + ".bias"] = layer.bias.detach().cpu().clone()
    return saved


def restore_original_weights(model, saved):
    with torch.no_grad():
        for name, layer in get_prunable_linear_layers(model):
            layer.weight.copy_(saved[name + ".weight"])
            layer.bias.copy_(saved[name + ".bias"])


# ---------------------------------------------------------------------------
# Fisher information
# ---------------------------------------------------------------------------

def sample_calibration_windows(train_tokens, num_samples, seq_len, seed):
    generator = torch.Generator().manual_seed(seed)
    max_start = train_tokens.shape[0] - seq_len - 1
    starts = torch.randint(0, max_start, (num_samples,), generator=generator)
    windows = []
    for index in range(num_samples):
        start = starts[index].item()
        windows.append(train_tokens[start:start + seq_len])
    return windows


def compute_fisher_diagonal(model, calibration_windows, device):
    """
    Empirical Fisher, used as a Hessian approximation for OBD.

    fisher[layer]: F_ii = E[(dL/dw_i)^2] for every weight.
    gate_fisher:   Fisher of a virtual 0/1 gate on each head and MLP neuron output,
                   E[(sum of g*w over the unit's output weights)^2]. Summing before squaring
                   keeps the interactions between a unit's weights.
    """
    layers = get_prunable_linear_layers(model)
    fisher = {}
    for name, layer in layers:
        fisher[name] = torch.zeros_like(layer.weight)
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attention.num_heads
    head_dim = model.blocks[0].attention.head_dim
    d_mlp = model.blocks[0].mlp.fc_in.weight.shape[0]
    gate_fisher = {
        "heads": torch.zeros(num_layers, num_heads, device=device),
        "neurons": torch.zeros(num_layers, d_mlp, device=device),
    }

    model.eval()
    for index in range(len(calibration_windows)):
        input_ids = calibration_windows[index].unsqueeze(0).to(device)
        model.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits[0, :-1, :], input_ids[0, 1:])
        loss.backward()
        with torch.no_grad():
            for name, layer in layers:
                fisher[name] += layer.weight.grad ** 2

            for block_index in range(num_layers):
                block = model.blocks[block_index]
                out_proj = block.attention.out_proj
                per_column = (out_proj.weight.grad * out_proj.weight).sum(dim=0)
                per_head = per_column.view(num_heads, head_dim).sum(dim=1)
                gate_fisher["heads"][block_index] += per_head ** 2

                fc_out = block.mlp.fc_out
                per_neuron = (fc_out.weight.grad * fc_out.weight).sum(dim=0)
                gate_fisher["neurons"][block_index] += per_neuron ** 2
        if (index + 1) % 32 == 0:
            print("  Fisher: processed %d / %d samples" % (index + 1, len(calibration_windows)))

    model.zero_grad(set_to_none=True)
    for name, layer in layers:
        fisher[name] /= len(calibration_windows)
    gate_fisher["heads"] /= len(calibration_windows)
    gate_fisher["neurons"] /= len(calibration_windows)
    return fisher, gate_fisher


# ---------------------------------------------------------------------------
# Scores and masks
# ---------------------------------------------------------------------------

def weight_scores(layer, method, fisher_for_layer):
    """magnitude: |w|.  obd: 0.5 * F_ii * w^2 (estimated loss increase from zeroing w)."""
    weight = layer.weight.detach()
    if method == "magnitude":
        return weight.abs()
    if method == "obd":
        return 0.5 * fisher_for_layer * weight * weight
    raise ValueError("unknown method: " + method)


def unstructured_mask(scores, sparsity):
    flat_scores = scores.flatten()
    num_to_prune = int(round(flat_scores.numel() * sparsity))
    mask = torch.ones_like(flat_scores)
    if num_to_prune > 0:
        order = torch.argsort(flat_scores)
        mask[order[:num_to_prune]] = 0.0
    return mask.view_as(scores)


def nm_mask(scores, n_keep, m):
    """Keep the n_keep highest-scoring weights in every group of m along the input dimension."""
    num_rows = scores.shape[0]
    num_columns = scores.shape[1]
    groups = scores.reshape(num_rows, num_columns // m, m)
    keep_indices = torch.topk(groups, n_keep, dim=-1).indices
    mask = torch.zeros_like(groups)
    mask.scatter_(-1, keep_indices, 1.0)
    return mask.reshape(num_rows, num_columns)


def apply_mask(layer, mask):
    with torch.no_grad():
        layer.weight.mul_(mask)


# ---------------------------------------------------------------------------
# Structured pruning (heads and MLP neurons)
# ---------------------------------------------------------------------------

def head_scores(block, block_index, method, fisher, gate_fisher):
    """
    magnitude: sum of w^2 over the head's Q/K/V rows and out_proj columns
    obd_naive: sum of 0.5*F_ii*w^2 over the same weights
    obd:       0.5 * gate Fisher of the head
    """
    if method == "obd":
        return 0.5 * gate_fisher["heads"][block_index]
    attention = block.attention
    head_dim = attention.head_dim
    prefix = "blocks." + str(block_index) + ".attention."
    scores = torch.zeros(attention.num_heads, device=attention.q_proj.weight.device)

    projection_names = ["q_proj", "k_proj", "v_proj"]
    for head in range(attention.num_heads):
        start = head * head_dim
        end = start + head_dim
        total = 0.0
        for projection_name in projection_names:
            layer = getattr(attention, projection_name)
            weight = layer.weight.detach()[start:end, :]
            if method == "magnitude":
                total = total + (weight * weight).sum()
            else:
                fisher_part = fisher[prefix + projection_name][start:end, :]
                total = total + (0.5 * fisher_part * weight * weight).sum()
        out_weight = attention.out_proj.weight.detach()[:, start:end]
        if method == "magnitude":
            total = total + (out_weight * out_weight).sum()
        else:
            fisher_part = fisher[prefix + "out_proj"][:, start:end]
            total = total + (0.5 * fisher_part * out_weight * out_weight).sum()
        scores[head] = total
    return scores


def remove_heads(block, heads_to_remove):
    attention = block.attention
    head_dim = attention.head_dim
    with torch.no_grad():
        for head in heads_to_remove:
            start = head * head_dim
            end = start + head_dim
            for layer in [attention.q_proj, attention.k_proj, attention.v_proj]:
                layer.weight[start:end, :] = 0.0
                layer.bias[start:end] = 0.0
            attention.out_proj.weight[:, start:end] = 0.0


def neuron_scores(block, block_index, method, fisher, gate_fisher):
    """Same three methods as head_scores; a neuron owns one fc_in row and one fc_out column."""
    if method == "obd":
        return 0.5 * gate_fisher["neurons"][block_index]
    fc_in_weight = block.mlp.fc_in.weight.detach()
    fc_out_weight = block.mlp.fc_out.weight.detach()
    if method == "magnitude":
        in_part = (fc_in_weight * fc_in_weight).sum(dim=1)
        out_part = (fc_out_weight * fc_out_weight).sum(dim=0)
    else:
        prefix = "blocks." + str(block_index) + ".mlp."
        in_part = (0.5 * fisher[prefix + "fc_in"] * fc_in_weight * fc_in_weight).sum(dim=1)
        out_part = (0.5 * fisher[prefix + "fc_out"] * fc_out_weight * fc_out_weight).sum(dim=0)
    return in_part + out_part


def remove_neurons(block, neurons_to_remove):
    with torch.no_grad():
        block.mlp.fc_in.weight[neurons_to_remove, :] = 0.0
        block.mlp.fc_in.bias[neurons_to_remove] = 0.0
        block.mlp.fc_out.weight[:, neurons_to_remove] = 0.0


def lowest_indices(scores, count):
    order = torch.argsort(scores)
    return order[:count]


def prune_structured_block(block, block_index, method, fisher, gate_fisher,
                           head_sparsity, neuron_sparsity):
    num_heads_to_remove = int(round(block.attention.num_heads * head_sparsity))
    if num_heads_to_remove > 0:
        scores = head_scores(block, block_index, method, fisher, gate_fisher)
        remove_heads(block, lowest_indices(scores, num_heads_to_remove).tolist())

    num_neurons = block.mlp.fc_in.weight.shape[0]
    num_neurons_to_remove = int(round(num_neurons * neuron_sparsity))
    if num_neurons_to_remove > 0:
        scores = neuron_scores(block, block_index, method, fisher, gate_fisher)
        remove_neurons(block, lowest_indices(scores, num_neurons_to_remove))


# ---------------------------------------------------------------------------
# Whole-model pruning
# ---------------------------------------------------------------------------

def prune_layer(layer, method, granularity, sparsity, fisher_for_layer, n_keep=None, m=None):
    scores = weight_scores(layer, method, fisher_for_layer)
    if granularity == "unstructured":
        mask = unstructured_mask(scores, sparsity)
    elif granularity == "nm":
        mask = nm_mask(scores, n_keep, m)
    else:
        raise ValueError("prune_layer handles 'unstructured' and 'nm' only")
    apply_mask(layer, mask)


def prune_model_uniform(model, method, granularity, sparsity, fisher, gate_fisher=None,
                        n_keep=None, m=None):
    """Prune every block to the same sparsity (unstructured, n_keep:m, or structured)."""
    if granularity == "structured":
        for block_index in range(len(model.blocks)):
            prune_structured_block(model.blocks[block_index], block_index, method, fisher,
                                   gate_fisher, head_sparsity=sparsity, neuron_sparsity=sparsity)
        return
    for name, layer in get_prunable_linear_layers(model):
        fisher_for_layer = None
        if fisher is not None:
            fisher_for_layer = fisher[name]
        prune_layer(layer, method, granularity, sparsity, fisher_for_layer, n_keep, m)


def measure_sparsity(model):
    """Fraction of block Linear weights that are exactly zero."""
    num_zero = 0
    num_total = 0
    for name, layer in get_prunable_linear_layers(model):
        num_zero = num_zero + (layer.weight == 0).sum().item()
        num_total = num_total + layer.weight.numel()
    return num_zero / num_total
