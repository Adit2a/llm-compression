"""WikiText-2 loading and perplexity evaluation."""

import os
import math

import torch
import torch.nn.functional as F

from common import PROJECT_DIR

DATA_DIR = os.path.join(PROJECT_DIR, "data")


def load_wikitext2_tokens(split):
    """Return a WikiText-2 split as one 1-D tensor of GPT-2 token ids (cached in data/)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "wikitext2_" + split + "_tokens.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    print("Downloading and tokenizing WikiText-2 (" + split + ")...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    token_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    torch.save(token_ids, cache_path)
    print("WikiText-2", split, "has", token_ids.shape[0], "tokens.")
    return token_ids


def evaluate_perplexity(model, token_ids, device, seq_len=1024, stride=None, max_windows=None):
    """
    Perplexity over a long token stream.
    stride = seq_len gives non-overlapping windows; stride < seq_len gives a sliding
    window where each token is scored once but with extra left context.
    max_windows limits evaluation to the first N windows.
    """
    if stride is None:
        stride = seq_len
    model.eval()
    num_tokens = token_ids.shape[0]
    total_loss = 0.0
    total_scored = 0
    previous_end = 0
    num_windows = 0

    with torch.no_grad():
        for begin in range(0, num_tokens, stride):
            end = min(begin + seq_len, num_tokens)
            if end - begin < 2:
                break
            input_ids = token_ids[begin:end].unsqueeze(0).to(device)
            logits = model(input_ids)

            predicted_logits = logits[0, :-1, :].float()
            correct_next_tokens = input_ids[0, 1:]
            losses = F.cross_entropy(predicted_logits, correct_next_tokens, reduction="none")

            # only score tokens that the previous window has not scored
            num_new_tokens = end - previous_end
            num_to_score = min(num_new_tokens, losses.shape[0])
            total_loss = total_loss + losses[-num_to_score:].sum().item()
            total_scored = total_scored + num_to_score

            previous_end = end
            num_windows = num_windows + 1
            if end == num_tokens:
                break
            if max_windows is not None and num_windows >= max_windows:
                break

    return math.exp(total_loss / total_scored)
