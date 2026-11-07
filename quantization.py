"""
Weight quantization: round-to-nearest (RTN) and AWQ.

Quantization is simulated: weights are rounded to the values an INT8/INT4 format can
represent and stored back as floats, so accuracy matches a real low-bit kernel.
Format: group-wise (128 weights per group) asymmetric, FP16 scale and zero point per group.
"""

import torch

from gpt2_model import get_prunable_linear_layers


def quantize_weight(weight, bits, group_size):
    """Group-wise asymmetric quantize + dequantize. Exact zeros stay zero (integer zero point)."""
    num_rows = weight.shape[0]
    num_columns = weight.shape[1]
    max_level = 2 ** bits - 1
    groups = weight.reshape(num_rows, num_columns // group_size, group_size)

    group_min = groups.amin(dim=-1, keepdim=True)
    group_max = groups.amax(dim=-1, keepdim=True)
    group_min = torch.clamp(group_min, max=0.0)
    group_max = torch.clamp(group_max, min=0.0)
    scale = (group_max - group_min) / max_level
    scale = torch.clamp(scale, min=1e-8)
    zero_point = torch.round(-group_min / scale)

    q = torch.clamp(torch.round(groups / scale) + zero_point, 0, max_level)
    dequantized = (q - zero_point) * scale
    return dequantized.reshape(num_rows, num_columns)


def quantize_model_rtn(model, bits, group_size):
    with torch.no_grad():
        for name, layer in get_prunable_linear_layers(model):
            layer.weight.copy_(quantize_weight(layer.weight, bits, group_size))


# ---------------------------------------------------------------------------
# AWQ
# ---------------------------------------------------------------------------

class InputCatcher:
    """Forward hook that stores a Linear layer's inputs as (tokens, in_features)."""

    def __init__(self):
        self.saved_inputs = []

    def __call__(self, module, inputs, output):
        x = inputs[0].detach()
        self.saved_inputs.append(x.reshape(-1, x.shape[-1]))

    def all_inputs(self):
        return torch.cat(self.saved_inputs, dim=0)


def capture_block_inputs(model, block_index, calibration_windows, device):
    block = model.blocks[block_index]
    layers = {
        "q_proj": block.attention.q_proj, "k_proj": block.attention.k_proj,
        "v_proj": block.attention.v_proj, "out_proj": block.attention.out_proj,
        "fc_in": block.mlp.fc_in, "fc_out": block.mlp.fc_out,
    }
    catchers = {}
    handles = []
    for name in layers:
        catchers[name] = InputCatcher()
        handles.append(layers[name].register_forward_hook(catchers[name]))
    with torch.no_grad():
        for window in calibration_windows:
            model(window.unsqueeze(0).to(device))
    for handle in handles:
        handle.remove()

    inputs = {}
    for name in layers:
        inputs[name] = catchers[name].all_inputs()
    return layers, inputs


def awq_search_layer(weight, inputs, bits, group_size, alphas):
    """
    Search s = mean|x|^alpha per input channel. W x = (W*s)(x/s), so quantizing W*s
    gives channels with large activations more precision. Keeps the alpha with the lowest
    output MSE on the calibration inputs (alpha = 0 is plain RTN).
    Returns Q(W*s)/s, the chosen alpha and all errors.
    """
    activation_size = inputs.abs().mean(dim=0)
    reference_output = inputs @ weight.t()
    best_error = None
    best_alpha = None
    best_weight = None
    errors = []
    for alpha in alphas:
        s = activation_size.pow(alpha).clamp(min=1e-4)
        s = s / torch.sqrt(s.max() * s.min())
        effective_weight = quantize_weight(weight * s, bits, group_size) / s
        output = inputs @ effective_weight.t()
        error = ((output - reference_output) ** 2).mean().item()
        errors.append(error)
        if best_error is None or error < best_error:
            best_error = error
            best_alpha = alpha
            best_weight = effective_weight
    return best_weight, best_alpha, errors


def quantize_model_awq(model, bits, group_size, calibration_windows, device, num_alphas=20):
    """
    AWQ applied block by block. Differences from the paper: each Linear gets its own scale
    (the paper shares one across layers with the same input so 1/s can be folded into the
    previous LayerNorm), 1/s is applied to the weight instead of being folded, and there is
    no clipping search.
    """
    alphas = []
    for index in range(num_alphas + 1):
        alphas.append(index / num_alphas)

    chosen_alphas = {}
    for block_index in range(len(model.blocks)):
        layers, inputs = capture_block_inputs(model, block_index, calibration_windows, device)
        with torch.no_grad():
            for name in layers:
                layer = layers[name]
                new_weight, best_alpha, errors = awq_search_layer(
                    layer.weight.detach(), inputs[name], bits, group_size, alphas)
                layer.weight.copy_(new_weight)
                full_name = "blocks." + str(block_index) + "." + name
                chosen_alphas[full_name] = best_alpha
        del inputs
        torch.cuda.empty_cache()
    return chosen_alphas


def model_size_mb(model, block_weight_bits_per_weight):
    """Block Linear weights at the given bits per weight, everything else in FP16."""
    block_weight_count = 0
    for name, layer in get_prunable_linear_layers(model):
        block_weight_count = block_weight_count + layer.weight.numel()
    total_count = 0
    for parameter in model.parameters():
        total_count = total_count + parameter.numel()
    other_count = total_count - block_weight_count

    block_bytes = block_weight_count * block_weight_bits_per_weight / 8
    other_bytes = other_count * 2
    return (block_bytes + other_bytes) / 1024 / 1024
