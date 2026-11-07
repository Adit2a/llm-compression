"""Shared helpers: device selection, seeding, results folder."""

import os
import json
import random

import numpy as np
import torch


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU.")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def save_json(data, filename):
    results_dir = make_results_dir()
    path = os.path.join(results_dir, filename)
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=2)
    print("Saved", path)
    return path


def count_parameters(model):
    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total = total + parameter.numel()
    return total
