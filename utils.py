from typing import List

import torch
from safetensors import safe_open
from torch.nn import Module
import numpy as np

from typing import List, Optional
import os 

def load_checkpoint_old(model, ckpt_paths):
    for path in ckpt_paths:
        checkpoint = torch.load(path, map_location=model.device)
        model.load_state_dict(checkpoint, strict=False)
    return model


def load_safetensors(
    model: Module,
    model_path: str,
    filter_prefixs: List[str] = None,
    strict: bool = True,
) -> Module:
    """
    Loads a model's state_dict from a safetensors file and applies optional filtering to exclude specific keys.

    Args:
        model (Module): The PyTorch model to load the weights into.
        model_path (str): Path to the safetensors file.
        filter_prefixst (List[str], optional): A set of keys to filter out from the state_dict. Default is None.
        strict (bool): Whether to strictly enforce that the keys in the model's state_dict match the file. Default is True.

    Returns:
        model (Module): The model with the loaded weights.
    """
    try:
        with safe_open(model_path, framework="pt") as f:
            state_dict = {}
            for key in f.keys():
                if filter_prefixs is not None and any(f in key for f in filter_prefixs):
                    continue
                state_dict[key] = f.get_tensor(key)
        # TODO: log how many variables are loaded
        model.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        raise RuntimeError(f"Failed to load safetensors from {model_path}: {e}")

    return model
    
    
def load_checkpoint(
    model: torch.nn.Module,
    model_path: str,
    filter_prefixes: Optional[List[str]] = None,
    strict: bool = True,
) -> torch.nn.Module:
    """
    Load a model's state_dict from either a safetensors file or a PyTorch .bin file,
    with optional key filtering.

    Args:
        model: The PyTorch model to load the weights into.
        model_path: Path to the checkpoint file (.safetensors or .bin).
        filter_prefixes: List of prefixes; keys starting with any of these prefixes
            will be skipped. Default is None.
        strict: Whether to strictly enforce that the keys in the model's state_dict
            match the file. Default is True.

    Returns:
        The model with the loaded weights.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    state_dict = {}

    try:
        if model_path.endswith(".safetensors"):
            # Old behavior: safetensors
            with safe_open(model_path, framework="pt") as f:
                for key in f.keys():
                    if filter_prefixes is not None and any(
                        key.startswith(p) for p in filter_prefixes
                    ):
                        continue
                    state_dict[key] = f.get_tensor(key)
        else:
            # New behavior: PyTorch .bin or any torch.save format
            raw_sd = torch.load(model_path, map_location="cpu")

            # In case the checkpoint is a dict with a 'state_dict' key (common pattern)
            if isinstance(raw_sd, dict) and "state_dict" in raw_sd:
                raw_sd = raw_sd["state_dict"]

            for key, tensor in raw_sd.items():
                if filter_prefixes is not None and any(
                    key.startswith(p) for p in filter_prefixes
                ):
                    continue
                state_dict[key] = tensor

        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        print(f"[load_checkpoint] Loaded {len(state_dict)} parameters from {model_path}")
        if missing:
            print(f"[load_checkpoint] Missing keys ({len(missing)}): {missing[:10]}...")
        if unexpected:
            print(f"[load_checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {model_path}: {e}")

    return model
