"""GPT-2 small (124M) implemented from scratch, plus a loader for the pretrained HuggingFace weights."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2Config:
    def __init__(self):
        self.vocab_size = 50257
        self.max_len = 1024
        self.d_model = 768
        self.num_layers = 12
        self.num_heads = 12
        self.d_mlp = 3072
        self.layer_norm_eps = 1e-5


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.num_heads

        # separate projections (HF fuses Q/K/V into one matrix; easier to prune this way)
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        causal_mask = torch.tril(torch.ones(config.max_len, config.max_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def split_heads(self, x):
        """(batch, seq, d_model) -> (batch, heads, seq, head_dim)"""
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def merge_heads(self, x):
        """(batch, heads, seq, head_dim) -> (batch, seq, d_model)"""
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, x):
        seq_len = x.shape[1]
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == False, float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)

        attended = weights @ v
        return self.out_proj(self.merge_heads(attended))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_in = nn.Linear(config.d_model, config.d_mlp)
        self.fc_out = nn.Linear(config.d_mlp, config.d_model)

    def forward(self, x):
        hidden = self.fc_in(x)
        # GPT-2 was trained with the tanh approximation of GELU
        hidden = F.gelu(hidden, approximate="tanh")
        return self.fc_out(hidden)


class Block(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.blocks = nn.ModuleList()
        for layer_index in range(config.num_layers):
            self.blocks.append(Block(config))
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def copy_linear_from_conv1d(linear, hf_weight, hf_bias):
    # HF's Conv1D stores weights as (in, out); nn.Linear uses (out, in)
    linear.weight.copy_(hf_weight.t())
    linear.bias.copy_(hf_bias)


def load_huggingface_weights(model, hf_state_dict):
    d_model = model.config.d_model
    with torch.no_grad():
        model.token_embedding.weight.copy_(hf_state_dict["transformer.wte.weight"])
        model.position_embedding.weight.copy_(hf_state_dict["transformer.wpe.weight"])

        for layer_index in range(model.config.num_layers):
            block = model.blocks[layer_index]
            prefix = "transformer.h." + str(layer_index) + "."

            block.norm1.weight.copy_(hf_state_dict[prefix + "ln_1.weight"])
            block.norm1.bias.copy_(hf_state_dict[prefix + "ln_1.bias"])
            block.norm2.weight.copy_(hf_state_dict[prefix + "ln_2.weight"])
            block.norm2.bias.copy_(hf_state_dict[prefix + "ln_2.bias"])

            # c_attn holds Q, K and V side by side: (768, 2304)
            qkv_weight = hf_state_dict[prefix + "attn.c_attn.weight"]
            qkv_bias = hf_state_dict[prefix + "attn.c_attn.bias"]
            q_weight = qkv_weight[:, 0:d_model]
            k_weight = qkv_weight[:, d_model:2 * d_model]
            v_weight = qkv_weight[:, 2 * d_model:3 * d_model]
            q_bias = qkv_bias[0:d_model]
            k_bias = qkv_bias[d_model:2 * d_model]
            v_bias = qkv_bias[2 * d_model:3 * d_model]
            copy_linear_from_conv1d(block.attention.q_proj, q_weight, q_bias)
            copy_linear_from_conv1d(block.attention.k_proj, k_weight, k_bias)
            copy_linear_from_conv1d(block.attention.v_proj, v_weight, v_bias)
            copy_linear_from_conv1d(block.attention.out_proj,
                                    hf_state_dict[prefix + "attn.c_proj.weight"],
                                    hf_state_dict[prefix + "attn.c_proj.bias"])

            copy_linear_from_conv1d(block.mlp.fc_in,
                                    hf_state_dict[prefix + "mlp.c_fc.weight"],
                                    hf_state_dict[prefix + "mlp.c_fc.bias"])
            copy_linear_from_conv1d(block.mlp.fc_out,
                                    hf_state_dict[prefix + "mlp.c_proj.weight"],
                                    hf_state_dict[prefix + "mlp.c_proj.bias"])

        model.final_norm.weight.copy_(hf_state_dict["transformer.ln_f.weight"])
        model.final_norm.bias.copy_(hf_state_dict["transformer.ln_f.bias"])


def load_pretrained_gpt2(device):
    from transformers import GPT2LMHeadModel

    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = GPT2(GPT2Config())
    load_huggingface_weights(model, hf_model.state_dict())
    del hf_model
    model = model.to(device)
    model.eval()
    return model


def get_prunable_linear_layers(model):
    """(name, layer) for the 72 Linear layers inside the blocks. Embeddings / lm_head are excluded."""
    layers = []
    for layer_index in range(model.config.num_layers):
        block = model.blocks[layer_index]
        prefix = "blocks." + str(layer_index) + "."
        layers.append((prefix + "attention.q_proj", block.attention.q_proj))
        layers.append((prefix + "attention.k_proj", block.attention.k_proj))
        layers.append((prefix + "attention.v_proj", block.attention.v_proj))
        layers.append((prefix + "attention.out_proj", block.attention.out_proj))
        layers.append((prefix + "mlp.fc_in", block.mlp.fc_in))
        layers.append((prefix + "mlp.fc_out", block.mlp.fc_out))
    return layers
