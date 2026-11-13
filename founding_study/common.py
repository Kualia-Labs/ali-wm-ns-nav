"""Shared helpers: load a sibling environment from its file and run the evaluation
protocol used for every run in this study."""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys

import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# condition -> environment file (numbers are the original platform row ids)
ENV_FILES = {
    "full": "envs/env_43_full.py",          # L1-L4 ("treatment")
    "baseline": "envs/env_44_baseline.py",  # L1 only
    "control": "envs/env_45_control.py",    # L1-L3, L4 zeroed
}


def load_env_class(condition: str):
    """Return the gymnasium.Env subclass defined in the condition's file."""
    path = os.path.join(HERE, ENV_FILES[condition])
    name = f"sibling_env_{condition}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, gym.Env) and obj is not gym.Env and obj.__module__ == name:
            return obj
    raise RuntimeError(f"no gymnasium.Env subclass in {path}")


def make_env(condition: str, rho: float | None = None):
    """Build the env the way the platform did: no arguments when rho is None
    (each sibling then uses its own default), otherwise env_class(rho=rho)."""
    cls = load_env_class(condition)
    return cls() if rho is None else cls(rho=rho)


def evaluate(model, env, n_episodes: int = 10, max_steps: int = 5000):
    """Deterministic evaluation on the training env object, episode i reset with seed=i.

    Success follows the platform rule: info["is_success"], else info["success"],
    else (episode return > 0). The sibling envs expose neither key, so success
    means a positive episode return.
    """
    returns, lengths, successes = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        total, steps = 0.0, 0
        done = truncated = False
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        while not (done or truncated) and steps < max_steps:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            episode_start = np.zeros((1,), dtype=bool)
            obs, reward, done, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
        returns.append(total)
        lengths.append(steps)
        successes.append(bool(info.get("is_success", info.get("success", total > 0))))
    return returns, lengths, successes
