"""
Phase 1: MLP and RNN from scratch, and experiments on why transformers replaced RNNs.

  A.  Linear model vs MLP on a three-spirals dataset
  B.  Hand-written RNN loop checked against nn.RNN
  C1. First-token recall: RNN vs one-layer self-attention at increasing lengths
  C2. Gradient reaching each input position (vanishing gradients)
  C3. Forward-pass speed, RNN vs attention

Usage: python phase1_foundations.py
"""

import math
import time

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import get_device, set_seed, make_results_dir, save_json, count_parameters


# ---------------------------------------------------------------------------
# Part A: MLP
# ---------------------------------------------------------------------------

def make_spiral_data(points_per_class, num_classes, noise):
    all_points = []
    all_labels = []
    for class_index in range(num_classes):
        radius = torch.linspace(0.0, 1.0, points_per_class)
        angle = torch.linspace(class_index * 4.0, (class_index + 1) * 4.0, points_per_class)
        angle = angle + torch.randn(points_per_class) * noise
        x = radius * torch.sin(angle)
        y = radius * torch.cos(angle)
        points = torch.stack([x, y], dim=1)
        labels = torch.full((points_per_class,), class_index, dtype=torch.long)
        all_points.append(points)
        all_labels.append(labels)
    points = torch.cat(all_points, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return points, labels


def split_train_test(points, labels, train_fraction):
    num_points = points.shape[0]
    permutation = torch.randperm(num_points)
    points = points[permutation]
    labels = labels[permutation]
    num_train = int(num_points * train_fraction)
    return points[:num_train], labels[:num_train], points[num_train:], labels[num_train:]


class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layer = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.layer(x)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden = torch.relu(self.layer1(x))
        hidden = torch.relu(self.layer2(hidden))
        return self.layer3(hidden)


def compute_accuracy(model, points, labels):
    model.eval()
    with torch.no_grad():
        predictions = model(points).argmax(dim=1)
        num_correct = (predictions == labels).sum().item()
    model.train()
    return num_correct / labels.shape[0]


def train_classifier(model, train_points, train_labels, num_steps, learning_rate):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    for step in range(num_steps):
        logits = model(train_points)
        loss = loss_function(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return loss_history


def plot_decision_boundaries(models, model_names, points, labels, device, path):
    grid_values = torch.linspace(-1.2, 1.2, 300)
    grid_x, grid_y = torch.meshgrid(grid_values, grid_values, indexing="xy")
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)

    figure, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    for index in range(len(models)):
        model = models[index]
        model.eval()
        with torch.no_grad():
            grid_predictions = model(grid_points).argmax(dim=1).cpu()
        model.train()
        axis = axes[index]
        axis.contourf(grid_x.numpy(), grid_y.numpy(),
                      grid_predictions.reshape(300, 300).numpy(),
                      alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5], cmap="viridis")
        axis.scatter(points[:, 0].cpu(), points[:, 1].cpu(), c=labels.cpu(),
                     s=8, cmap="viridis")
        axis.set_title(model_names[index])
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print("Saved", path)


def run_part_a(device, results_dir):
    print("\n========== PART A: MLP on three spirals ==========")
    set_seed(0)
    points, labels = make_spiral_data(points_per_class=300, num_classes=3, noise=0.2)
    train_points, train_labels, test_points, test_labels = split_train_test(points, labels, 0.8)
    train_points = train_points.to(device)
    train_labels = train_labels.to(device)
    test_points = test_points.to(device)
    test_labels = test_labels.to(device)

    linear_model = LinearClassifier(input_size=2, num_classes=3).to(device)
    mlp_model = MLP(input_size=2, hidden_size=64, num_classes=3).to(device)

    linear_losses = train_classifier(linear_model, train_points, train_labels,
                                     num_steps=2000, learning_rate=0.01)
    mlp_losses = train_classifier(mlp_model, train_points, train_labels,
                                  num_steps=2000, learning_rate=0.01)

    linear_accuracy = compute_accuracy(linear_model, test_points, test_labels)
    mlp_accuracy = compute_accuracy(mlp_model, test_points, test_labels)
    print("Linear model: %d params, test accuracy %.1f%%"
          % (count_parameters(linear_model), linear_accuracy * 100))
    print("MLP model:    %d params, test accuracy %.1f%%"
          % (count_parameters(mlp_model), mlp_accuracy * 100))

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(linear_losses, label="Linear (no hidden layer)")
    axis.plot(mlp_losses, label="MLP (2 hidden layers + ReLU)")
    axis.set_xlabel("training step")
    axis.set_ylabel("cross-entropy loss")
    axis.set_title("Training loss on three spirals")
    axis.legend()
    figure.tight_layout()
    loss_path = results_dir + "/phase1_mlp_loss.png"
    figure.savefig(loss_path, dpi=120)
    plt.close(figure)
    print("Saved", loss_path)

    plot_decision_boundaries(
        [linear_model, mlp_model],
        ["Linear: %.0f%% test accuracy" % (linear_accuracy * 100),
         "MLP: %.0f%% test accuracy" % (mlp_accuracy * 100)],
        test_points, test_labels, device,
        results_dir + "/phase1_mlp_decision_boundary.png")

    return {
        "linear_test_accuracy": linear_accuracy,
        "mlp_test_accuracy": mlp_accuracy,
        "linear_final_loss": linear_losses[-1],
        "mlp_final_loss": mlp_losses[-1],
        "linear_num_params": count_parameters(linear_model),
        "mlp_num_params": count_parameters(mlp_model),
    }


# ---------------------------------------------------------------------------
# Part B: RNN
# ---------------------------------------------------------------------------

class ManualRNN(nn.Module):
    """h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        hidden = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        all_hidden = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            hidden = torch.tanh(self.input_to_hidden(x_t) + self.hidden_to_hidden(hidden))
            all_hidden.append(hidden)
        return torch.stack(all_hidden, dim=1), hidden


def run_part_b(device):
    print("\n========== PART B: hand-written RNN vs nn.RNN ==========")
    set_seed(0)
    input_size = 16
    hidden_size = 32
    manual_rnn = ManualRNN(input_size, hidden_size).to(device)
    pytorch_rnn = nn.RNN(input_size, hidden_size, batch_first=True).to(device)

    with torch.no_grad():
        pytorch_rnn.weight_ih_l0.copy_(manual_rnn.input_to_hidden.weight)
        pytorch_rnn.bias_ih_l0.copy_(manual_rnn.input_to_hidden.bias)
        pytorch_rnn.weight_hh_l0.copy_(manual_rnn.hidden_to_hidden.weight)
        pytorch_rnn.bias_hh_l0.copy_(manual_rnn.hidden_to_hidden.bias)

    inputs = torch.randn(4, 50, input_size, device=device)
    with torch.no_grad():
        manual_outputs, manual_last = manual_rnn(inputs)
        pytorch_outputs, pytorch_last = pytorch_rnn(inputs)
    max_difference = (manual_outputs - pytorch_outputs).abs().max().item()
    print("Max absolute difference between hand-written RNN and nn.RNN: %.2e" % max_difference)
    if max_difference < 1e-5:
        print("MATCH")
    else:
        print("MISMATCH")
    return {"manual_vs_pytorch_rnn_max_abs_diff": max_difference}


# ---------------------------------------------------------------------------
# Part C: RNN vs attention
# ---------------------------------------------------------------------------

def make_recall_batch(batch_size, seq_len, vocab_size, device):
    """Random digit sequences; the label is the first digit."""
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = tokens[:, 0].clone()
    return tokens, labels


class RecallRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward_from_embeddings(self, embedded):
        all_hidden, last_hidden = self.rnn(embedded)
        return self.output_layer(all_hidden[:, -1, :])

    def forward(self, tokens):
        return self.forward_from_embeddings(self.embedding(tokens))


class TinyAttentionClassifier(nn.Module):
    """One layer of single-head self-attention; predicts from the last position."""

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def attention(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = q @ k.transpose(1, 2) / math.sqrt(self.embed_dim)
        weights = torch.softmax(scores, dim=-1)
        return weights @ v

    def forward_from_embeddings(self, embedded):
        seq_len = embedded.shape[1]
        positions = torch.arange(seq_len, device=embedded.device)
        x = embedded + self.position_embedding(positions)
        x = x + self.attention(x)
        return self.output_layer(x[:, -1, :])

    def forward(self, tokens):
        return self.forward_from_embeddings(self.token_embedding(tokens))


def train_recall_model(model, seq_len, vocab_size, device, num_steps, batch_size, learning_rate):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for step in range(num_steps):
        tokens, labels = make_recall_batch(batch_size, seq_len, vocab_size, device)
        logits = model(tokens)
        loss = loss_function(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return loss.item()


def evaluate_recall_model(model, seq_len, vocab_size, device, num_examples):
    model.eval()
    with torch.no_grad():
        tokens, labels = make_recall_batch(num_examples, seq_len, vocab_size, device)
        predictions = model(tokens).argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
    model.train()
    return accuracy


def run_part_c1(device, results_dir):
    print("\n========== PART C1: first-token recall vs sequence length ==========")
    vocab_size = 10
    sequence_lengths = [5, 20, 50, 100]
    rnn_accuracies = []
    attention_accuracies = []
    for seq_len in sequence_lengths:
        set_seed(0)
        rnn_model = RecallRNN(vocab_size, embed_dim=32, hidden_size=64).to(device)
        start = time.time()
        train_recall_model(rnn_model, seq_len, vocab_size, device,
                           num_steps=2000, batch_size=128, learning_rate=0.001)
        rnn_accuracy = evaluate_recall_model(rnn_model, seq_len, vocab_size, device, 4000)
        rnn_time = time.time() - start

        set_seed(0)
        attention_model = TinyAttentionClassifier(vocab_size, embed_dim=64, max_len=128).to(device)
        start = time.time()
        train_recall_model(attention_model, seq_len, vocab_size, device,
                           num_steps=2000, batch_size=128, learning_rate=0.001)
        attention_accuracy = evaluate_recall_model(attention_model, seq_len, vocab_size, device, 4000)
        attention_time = time.time() - start

        print("seq_len %3d | RNN accuracy %5.1f%% (%.0fs) | Attention accuracy %5.1f%% (%.0fs)"
              % (seq_len, rnn_accuracy * 100, rnn_time, attention_accuracy * 100, attention_time))
        rnn_accuracies.append(rnn_accuracy)
        attention_accuracies.append(attention_accuracy)

    rnn_percent = []
    attention_percent = []
    for index in range(len(sequence_lengths)):
        rnn_percent.append(rnn_accuracies[index] * 100)
        attention_percent.append(attention_accuracies[index] * 100)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(sequence_lengths, rnn_percent, marker="o", label="RNN")
    axis.plot(sequence_lengths, attention_percent, marker="o", label="1-layer self-attention")
    axis.axhline(100.0 / vocab_size, color="gray", linestyle="--", label="random guessing (10%)")
    axis.set_xlabel("sequence length")
    axis.set_ylabel("test accuracy (%)")
    axis.set_title("Recall the FIRST token after reading the whole sequence")
    axis.set_ylim(0, 105)
    axis.legend()
    figure.tight_layout()
    path = results_dir + "/phase1_recall_vs_length.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print("Saved", path)

    return {
        "sequence_lengths": sequence_lengths,
        "rnn_accuracy": rnn_accuracies,
        "attention_accuracy": attention_accuracies,
    }


def gradient_norm_per_position(model, embedding_layer, seq_len, vocab_size, device):
    """Mean L2 norm of d(loss)/d(input embedding) at every position."""
    tokens, labels = make_recall_batch(256, seq_len, vocab_size, device)
    embedded = embedding_layer(tokens).detach()
    embedded.requires_grad_(True)
    logits = model.forward_from_embeddings(embedded)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    norms = embedded.grad.norm(dim=2).mean(dim=0)
    return norms.cpu().tolist()


def run_part_c2(device, results_dir):
    print("\n========== PART C2: vanishing gradients ==========")
    vocab_size = 10
    seq_len = 100
    set_seed(0)
    rnn_model = RecallRNN(vocab_size, embed_dim=32, hidden_size=64).to(device)
    attention_model = TinyAttentionClassifier(vocab_size, embed_dim=64, max_len=128).to(device)

    # float64: in float32 the RNN gradient at position 0 underflows to exactly 0
    rnn_model = rnn_model.double()
    attention_model = attention_model.double()

    rnn_norms = gradient_norm_per_position(rnn_model, rnn_model.embedding, seq_len, vocab_size, device)
    attention_norms = gradient_norm_per_position(attention_model, attention_model.token_embedding,
                                                 seq_len, vocab_size, device)

    print("position | RNN gradient | attention gradient")
    for position in [0, 10, 25, 50, 75, 90, 99]:
        print("   %3d   |   %.2e   |   %.2e" % (position, rnn_norms[position], attention_norms[position]))

    figure, axis = plt.subplots(figsize=(7, 4))
    positions = list(range(seq_len))
    axis.semilogy(positions, rnn_norms, label="RNN")
    axis.semilogy(positions, attention_norms, label="1-layer self-attention")
    axis.set_xlabel("input position (0 = first token, the one we must recall)")
    axis.set_ylabel("gradient size (log scale)")
    axis.set_title("How much gradient reaches each input position (untrained models)")
    axis.legend()
    figure.tight_layout()
    path = results_dir + "/phase1_gradient_vs_position.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print("Saved", path)

    return {
        "rnn_grad_first_position": rnn_norms[0],
        "rnn_grad_last_position": rnn_norms[-1],
        "attention_grad_first_position": attention_norms[0],
        "attention_grad_second_last_position": attention_norms[-2],
    }


def time_forward(function, inputs, num_repeats):
    """Average milliseconds per call."""
    with torch.no_grad():
        for i in range(3):
            function(inputs)
        if inputs.is_cuda:
            torch.cuda.synchronize()
        start = time.time()
        for i in range(num_repeats):
            function(inputs)
        if inputs.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - start
    return elapsed / num_repeats * 1000.0


def run_part_c3(device):
    print("\n========== PART C3: speed, sequential RNN vs parallel attention ==========")
    set_seed(0)
    batch_size = 16
    sequence_lengths = [64, 256, 1024, 2048]
    results = {"batch_size": batch_size, "sequence_lengths": sequence_lengths}
    # width 128: cuDNN's fused RNN kernel is very fast; width 768 matches GPT-2
    for dim in [128, 768]:
        rnn = nn.RNN(dim, dim, batch_first=True).to(device)
        attention_model = TinyAttentionClassifier(vocab_size=10, embed_dim=dim, max_len=4096).to(device)
        rnn_times = []
        attention_times = []
        for seq_len in sequence_lengths:
            inputs = torch.randn(batch_size, seq_len, dim, device=device)
            rnn_ms = time_forward(rnn, inputs, num_repeats=10)
            attention_ms = time_forward(attention_model.attention, inputs, num_repeats=10)
            print("dim %4d | seq_len %5d | RNN %7.2f ms | attention %7.2f ms"
                  % (dim, seq_len, rnn_ms, attention_ms))
            rnn_times.append(rnn_ms)
            attention_times.append(attention_ms)
        results["dim_%d_rnn_forward_ms" % dim] = rnn_times
        results["dim_%d_attention_forward_ms" % dim] = attention_times
        del rnn, attention_model
        torch.cuda.empty_cache()
    return results


def main():
    device = get_device()
    results_dir = make_results_dir()
    results = {}
    results["part_a_mlp"] = run_part_a(device, results_dir)
    results["part_b_rnn_check"] = run_part_b(device)
    results["part_c1_recall"] = run_part_c1(device, results_dir)
    results["part_c2_gradients"] = run_part_c2(device, results_dir)
    results["part_c3_speed"] = run_part_c3(device)
    save_json(results, "phase1_results.json")


if __name__ == "__main__":
    main()
