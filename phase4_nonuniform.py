"""
Phase 4: non-uniform per-layer sparsity at a fixed overall budget.
Compares uniform, NetAdapt-style greedy search and a simplified AMC (DDPG) agent.

Sparsity is allocated to 24 groups (attention and MLP of each block); inside a group
every Linear is pruned with unstructured OBD. Search uses the validation split,
final numbers use the test split.

Usage: python phase4_nonuniform.py [--quick]
"""

import sys
import math
import time
import random

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import get_device, set_seed, make_results_dir, save_json
from gpt2_model import load_pretrained_gpt2, get_prunable_linear_layers
from evaluation import load_wikitext2_tokens, evaluate_perplexity
from pruning import sample_calibration_windows, compute_fisher_diagonal, weight_scores


QUICK_MODE = "--quick" in sys.argv

SCORE_METHOD = "obd"
TARGET_SPARSITY = 0.5
MAX_GROUP_SPARSITY = 0.9
NUM_FISHER_SAMPLES = 128
FISHER_SEQ_LEN = 512

if QUICK_MODE:
    SEARCH_WINDOWS = 2
    SENSITIVITY_LEVELS = [0.5, 0.9]
    NETADAPT_STEP = 0.05
    AMC_EPISODES = 12
    AMC_WARMUP_EPISODES = 4
    FINAL_TEST_WINDOWS = 10
else:
    SEARCH_WINDOWS = 4
    SENSITIVITY_LEVELS = [0.3, 0.5, 0.7, 0.9]
    # fraction of all block weights removed per iteration
    # (+45% for one attention group, +22% for one MLP group)
    NETADAPT_STEP = 0.0125
    AMC_EPISODES = 300
    AMC_WARMUP_EPISODES = 60
    FINAL_TEST_WINDOWS = None


# ---------------------------------------------------------------------------
# Groups and pruner
# ---------------------------------------------------------------------------

def build_groups(model):
    layer_sizes = {}
    for name, layer in get_prunable_linear_layers(model):
        layer_sizes[name] = layer.weight.numel()
    groups = []
    for block_index in range(len(model.blocks)):
        prefix = "blocks." + str(block_index) + "."
        attention_layers = [prefix + "attention.q_proj", prefix + "attention.k_proj",
                            prefix + "attention.v_proj", prefix + "attention.out_proj"]
        mlp_layers = [prefix + "mlp.fc_in", prefix + "mlp.fc_out"]
        for kind, layer_names in [("attn", attention_layers), ("mlp", mlp_layers)]:
            num_params = 0
            for layer_name in layer_names:
                num_params = num_params + layer_sizes[layer_name]
            groups.append({"name": "b%d.%s" % (block_index, kind), "block": block_index,
                           "is_mlp": kind == "mlp", "layer_names": layer_names,
                           "num_params": num_params})
    return groups


class GroupPruner:
    """
    Scores are computed once; each layer keeps its weights sorted from least to most
    important, so setting a layer to sparsity s is just restore + zero the first s fraction.
    """

    def __init__(self, model, method, fisher):
        self.layers = {}
        self.original_weights = {}
        self.order = {}
        for name, layer in get_prunable_linear_layers(model):
            self.layers[name] = layer
            self.original_weights[name] = layer.weight.detach().clone()
            fisher_for_layer = None
            if fisher is not None:
                fisher_for_layer = fisher[name]
            scores = weight_scores(layer, method, fisher_for_layer).flatten()
            self.order[name] = torch.argsort(scores).to(torch.int32)

    def set_layer_sparsity(self, name, sparsity):
        layer = self.layers[name]
        with torch.no_grad():
            layer.weight.copy_(self.original_weights[name])
            num_to_prune = int(round(layer.weight.numel() * sparsity))
            if num_to_prune > 0:
                prune_indices = self.order[name][:num_to_prune].long()
                layer.weight.view(-1)[prune_indices] = 0.0

    def set_group_sparsity(self, group, sparsity):
        for name in group["layer_names"]:
            self.set_layer_sparsity(name, sparsity)

    def apply_allocation(self, groups, allocation):
        for index in range(len(groups)):
            self.set_group_sparsity(groups[index], allocation[index])


def overall_sparsity(groups, allocation):
    pruned = 0.0
    total = 0.0
    for index in range(len(groups)):
        pruned = pruned + allocation[index] * groups[index]["num_params"]
        total = total + groups[index]["num_params"]
    return pruned / total


# ---------------------------------------------------------------------------
# Sensitivity scan
# ---------------------------------------------------------------------------

def sensitivity_scan(model, pruner, groups, search_tokens, device, baseline_ppl, results_dir):
    print("\n========== STEP 1: per-group sensitivity (prune one group alone) ==========")
    num_groups = len(groups)
    table = []
    for group_index in range(num_groups):
        row = []
        for sparsity in SENSITIVITY_LEVELS:
            pruner.set_group_sparsity(groups[group_index], sparsity)
            perplexity = evaluate_perplexity(model, search_tokens, device, max_windows=SEARCH_WINDOWS)
            row.append(perplexity)
        pruner.set_group_sparsity(groups[group_index], 0.0)
        table.append(row)
        text = ""
        for level_index in range(len(SENSITIVITY_LEVELS)):
            text = text + "  %d%%: %8.2f" % (SENSITIVITY_LEVELS[level_index] * 100, row[level_index])
        print("%-8s%s" % (groups[group_index]["name"], text))

    values = []
    for row in table:
        value_row = []
        for perplexity in row:
            value_row.append(math.log(perplexity / baseline_ppl))
        values.append(value_row)
    figure, axis = plt.subplots(figsize=(6, 9))
    image = axis.imshow(values, aspect="auto", cmap="magma_r")
    axis.set_yticks(range(num_groups))
    group_names = []
    for group in groups:
        group_names.append(group["name"])
    axis.set_yticklabels(group_names, fontsize=8)
    level_names = []
    for sparsity in SENSITIVITY_LEVELS:
        level_names.append("%d%%" % (sparsity * 100))
    axis.set_xticks(range(len(SENSITIVITY_LEVELS)))
    axis.set_xticklabels(level_names)
    for group_index in range(num_groups):
        for level_index in range(len(SENSITIVITY_LEVELS)):
            axis.text(level_index, group_index, "%.1f" % table[group_index][level_index],
                      ha="center", va="center", fontsize=6, color="gray")
    axis.set_xlabel("sparsity of this ONE group (all others dense)")
    axis.set_title("Sensitivity: validation perplexity\n(colour = log(ppl / dense ppl))")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path = results_dir + "/phase4_sensitivity.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print("Saved", path)
    return table


# ---------------------------------------------------------------------------
# NetAdapt-style greedy search
# ---------------------------------------------------------------------------

def netadapt_search(model, pruner, groups, search_tokens, device):
    """
    Each iteration: for every group, try removing NETADAPT_STEP of all weights from it,
    evaluate, undo; keep the single best candidate. Repeat until the target is reached.
    Unlike the paper, the resource is weight count (not latency) and there is no fine-tuning.
    """
    print("\n========== STEP 3: NetAdapt-style greedy search ==========")
    num_groups = len(groups)
    total_params = 0
    for group in groups:
        total_params = total_params + group["num_params"]
    target_params = TARGET_SPARSITY * total_params

    allocation = [0.0] * num_groups
    pruner.apply_allocation(groups, allocation)
    history = []
    iteration = 0
    while True:
        pruned_params = overall_sparsity(groups, allocation) * total_params
        remaining = target_params - pruned_params
        if remaining <= 1:
            break
        step_params = min(NETADAPT_STEP * total_params, remaining)
        best_group = None
        best_ppl = None
        best_new_sparsity = None
        for group_index in range(num_groups):
            new_sparsity = allocation[group_index] + step_params / groups[group_index]["num_params"]
            if new_sparsity > MAX_GROUP_SPARSITY:
                continue
            pruner.set_group_sparsity(groups[group_index], new_sparsity)
            perplexity = evaluate_perplexity(model, search_tokens, device, max_windows=SEARCH_WINDOWS)
            pruner.set_group_sparsity(groups[group_index], allocation[group_index])
            if best_ppl is None or perplexity < best_ppl:
                best_ppl = perplexity
                best_group = group_index
                best_new_sparsity = new_sparsity
        if best_group is None:
            print("No group can take a full step any more - stopping early.")
            break
        allocation[best_group] = best_new_sparsity
        pruner.set_group_sparsity(groups[best_group], best_new_sparsity)
        iteration = iteration + 1
        current = overall_sparsity(groups, allocation)
        print("iteration %2d: pruned more from %-8s -> overall %.1f%%, validation ppl %.2f"
              % (iteration, groups[best_group]["name"], current * 100, best_ppl))
        history.append({"iteration": iteration, "group": groups[best_group]["name"],
                        "overall_sparsity": current, "search_ppl": best_ppl})
    return allocation, history


# ---------------------------------------------------------------------------
# Simplified AMC (DDPG)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, state):
        hidden = torch.relu(self.layer1(state))
        hidden = torch.relu(self.layer2(hidden))
        return torch.sigmoid(self.layer3(hidden)) * MAX_GROUP_SPARSITY


class Critic(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.layer1 = nn.Linear(state_size + 1, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, state, action):
        hidden = torch.relu(self.layer1(torch.cat([state, action], dim=1)))
        hidden = torch.relu(self.layer2(hidden))
        return self.layer3(hidden)


def make_state(group_index, groups, pruned_so_far, still_needed, previous_action, total_params, largest_group):
    group = groups[group_index]
    is_mlp = 0.0
    if group["is_mlp"]:
        is_mlp = 1.0
    return [
        group_index / (len(groups) - 1),
        is_mlp,
        group["num_params"] / largest_group,
        pruned_so_far / total_params,
        still_needed / total_params,
        previous_action,
    ]


def amc_episode(actor, groups, total_params, largest_group, noise_sigma, use_random_actions):
    """
    Choose a sparsity for each group in order. Actions are clipped so that the target
    is always reachable and never exceeded, so every episode ends at exactly the target.
    """
    target_params = TARGET_SPARSITY * total_params
    params_after = []
    running = 0
    for index in range(len(groups) - 1, -1, -1):
        params_after.insert(0, running)
        running = running + groups[index]["num_params"]

    states = []
    actions = []
    pruned_so_far = 0.0
    previous_action = 0.0
    for group_index in range(len(groups)):
        group_size = groups[group_index]["num_params"]
        still_needed = target_params - pruned_so_far
        state = make_state(group_index, groups, pruned_so_far, still_needed, previous_action,
                           total_params, largest_group)
        if use_random_actions:
            action = random.uniform(0.0, MAX_GROUP_SPARSITY)
        else:
            with torch.no_grad():
                action = actor(torch.tensor([state])).item()
            action = action + random.gauss(0.0, noise_sigma)
        lowest_allowed = (still_needed - MAX_GROUP_SPARSITY * params_after[group_index]) / group_size
        highest_allowed = still_needed / group_size
        lowest_allowed = max(0.0, lowest_allowed)
        highest_allowed = min(MAX_GROUP_SPARSITY, highest_allowed)
        action = min(max(action, lowest_allowed), highest_allowed)
        states.append(state)
        actions.append(action)
        pruned_so_far = pruned_so_far + action * group_size
        previous_action = action
    return states, actions


def amc_search(model, pruner, groups, search_tokens, device, results_dir):
    """
    Simplified AMC (He et al., 2018): DDPG-style actor-critic, per-group continuous action,
    budget enforced by clipping, random warm-up, decaying exploration noise, moving-average
    reward baseline. Simplifications: 24 groups instead of every layer, reward = -log(ppl),
    the critic regresses the final episode reward directly (no target networks), no fine-tuning.
    The best allocation seen during the search is returned.
    """
    print("\n========== STEP 4: simplified AMC (DDPG agent) ==========")
    total_params = 0
    largest_group = 0
    for group in groups:
        total_params = total_params + group["num_params"]
        largest_group = max(largest_group, group["num_params"])

    state_size = 6
    actor = Actor(state_size)
    critic = Critic(state_size)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    replay_states = []
    replay_actions = []
    replay_targets = []
    baseline = None
    noise_sigma = 0.5
    best_reward = None
    best_allocation = None
    reward_history = []

    for episode in range(AMC_EPISODES):
        use_random = episode < AMC_WARMUP_EPISODES
        states, actions = amc_episode(actor, groups, total_params, largest_group, noise_sigma, use_random)
        pruner.apply_allocation(groups, actions)
        perplexity = evaluate_perplexity(model, search_tokens, device, max_windows=SEARCH_WINDOWS)
        reward = -math.log(perplexity)
        reward_history.append(reward)

        if best_reward is None or reward > best_reward:
            best_reward = reward
            best_allocation = list(actions)

        if baseline is None:
            baseline = reward
        else:
            baseline = 0.95 * baseline + 0.05 * reward
        for index in range(len(states)):
            replay_states.append(states[index])
            replay_actions.append([actions[index]])
            replay_targets.append([reward - baseline])

        if not use_random:
            noise_sigma = noise_sigma * 0.985
            for update in range(20):
                batch_indices = random.sample(range(len(replay_states)), min(64, len(replay_states)))
                batch_states = []
                batch_actions = []
                batch_targets = []
                for index in batch_indices:
                    batch_states.append(replay_states[index])
                    batch_actions.append(replay_actions[index])
                    batch_targets.append(replay_targets[index])
                batch_states = torch.tensor(batch_states)
                batch_actions = torch.tensor(batch_actions)
                batch_targets = torch.tensor(batch_targets)

                predicted = critic(batch_states, batch_actions)
                critic_loss = ((predicted - batch_targets) ** 2).mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss = -critic(batch_states, actor(batch_states)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

        if (episode + 1) % 10 == 0 or episode == 0:
            mode = "policy"
            if use_random:
                mode = "random"
            print("episode %3d | %s | validation ppl %8.2f | best so far %8.2f | noise %.3f"
                  % (episode + 1, mode, perplexity, math.exp(-best_reward), noise_sigma))

    figure, axis = plt.subplots(figsize=(7, 4))
    perplexities = []
    for reward in reward_history:
        perplexities.append(math.exp(-reward))
    axis.plot(range(1, len(perplexities) + 1), perplexities, ".", alpha=0.5, label="episode")
    best_so_far = []
    current_best = None
    for value in perplexities:
        if current_best is None or value < current_best:
            current_best = value
        best_so_far.append(current_best)
    axis.plot(range(1, len(best_so_far) + 1), best_so_far, color="red", label="best so far")
    axis.axvline(AMC_WARMUP_EPISODES, color="gray", linestyle="--", label="end of random warm-up")
    axis.set_yscale("log")
    axis.set_xlabel("episode")
    axis.set_ylabel("validation perplexity (log)")
    axis.set_title("Simplified AMC search at %d%% overall sparsity" % (TARGET_SPARSITY * 100))
    axis.legend()
    figure.tight_layout()
    path = results_dir + "/phase4_amc_search.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print("Saved", path)
    return best_allocation, reward_history


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def plot_allocations(groups, allocations, results_dir):
    names = []
    for group in groups:
        names.append(group["name"])
    method_names = list(allocations.keys())
    bar_width = 0.27
    figure, axis = plt.subplots(figsize=(14, 4.5))
    for method_index in range(len(method_names)):
        x_positions = []
        heights = []
        for group_index in range(len(groups)):
            x_positions.append(group_index + (method_index - 1) * bar_width)
            heights.append(allocations[method_names[method_index]][group_index] * 100)
        axis.bar(x_positions, heights, width=bar_width, label=method_names[method_index])
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels(names, rotation=90, fontsize=8)
    axis.set_ylabel("sparsity of this group (%)")
    axis.set_title("Per-group sparsity at %d%% overall" % (TARGET_SPARSITY * 100))
    axis.legend()
    figure.tight_layout()
    path = results_dir + "/phase4_allocations.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print("Saved", path)


def main():
    set_seed(0)
    random.seed(0)
    device = get_device()
    results_dir = make_results_dir()
    if QUICK_MODE:
        print("QUICK MODE")

    model = load_pretrained_gpt2(device)
    test_tokens = load_wikitext2_tokens("test")
    validation_tokens = load_wikitext2_tokens("validation")
    train_tokens = load_wikitext2_tokens("train")
    groups = build_groups(model)

    fisher = None
    if SCORE_METHOD == "obd":
        print("Computing diagonal Fisher (%d train samples)..." % NUM_FISHER_SAMPLES)
        windows = sample_calibration_windows(train_tokens, NUM_FISHER_SAMPLES, FISHER_SEQ_LEN, seed=0)
        fisher, gate_fisher = compute_fisher_diagonal(model, windows, device)
        del gate_fisher
    pruner = GroupPruner(model, SCORE_METHOD, fisher)
    del fisher
    torch.cuda.empty_cache()

    dense_search_ppl = evaluate_perplexity(model, validation_tokens, device, max_windows=SEARCH_WINDOWS)
    print("Dense model, validation subset perplexity: %.2f" % dense_search_ppl)

    results = {"quick_mode": QUICK_MODE, "score_method": SCORE_METHOD,
               "target_sparsity": TARGET_SPARSITY, "search_windows": SEARCH_WINDOWS,
               "group_names": [], "group_sizes": []}
    for group in groups:
        results["group_names"].append(group["name"])
        results["group_sizes"].append(group["num_params"])
    output_name = "phase4_results.json"
    if QUICK_MODE:
        output_name = "phase4_results_quick.json"

    start = time.time()
    results["sensitivity_levels"] = SENSITIVITY_LEVELS
    results["sensitivity_ppl"] = sensitivity_scan(model, pruner, groups, validation_tokens, device,
                                                  dense_search_ppl, results_dir)
    results["sensitivity_seconds"] = time.time() - start
    save_json(results, output_name)

    uniform_allocation = [TARGET_SPARSITY] * len(groups)

    start = time.time()
    netadapt_allocation, netadapt_history = netadapt_search(model, pruner, groups, validation_tokens, device)
    results["netadapt_seconds"] = time.time() - start
    results["netadapt_history"] = netadapt_history
    save_json(results, output_name)

    start = time.time()
    amc_allocation, amc_rewards = amc_search(model, pruner, groups, validation_tokens, device, results_dir)
    results["amc_seconds"] = time.time() - start
    results["amc_rewards"] = amc_rewards

    print("\n========== STEP 5: final comparison on the WikiText-2 TEST set ==========")
    allocations = {"uniform": uniform_allocation, "NetAdapt": netadapt_allocation, "AMC (simplified)": amc_allocation}
    results["final"] = {}
    for method_name in allocations:
        allocation = allocations[method_name]
        pruner.apply_allocation(groups, allocation)
        search_ppl = evaluate_perplexity(model, validation_tokens, device, max_windows=SEARCH_WINDOWS)
        test_ppl = evaluate_perplexity(model, test_tokens, device, max_windows=FINAL_TEST_WINDOWS)
        sparsity = overall_sparsity(groups, allocation)
        print("%-17s | overall sparsity %.1f%% | validation-subset ppl %8.2f | TEST ppl %8.2f"
              % (method_name, sparsity * 100, search_ppl, test_ppl))
        results["final"][method_name] = {"allocation": allocation, "overall_sparsity": sparsity,
                                         "validation_subset_ppl": search_ppl, "test_ppl": test_ppl}
    pruner.apply_allocation(groups, [0.0] * len(groups))
    save_json(results, output_name)
    plot_allocations(groups, allocations, results_dir)


if __name__ == "__main__":
    main()
