"""Unit tests for src/utils.py — pins the strengthened, reproducible seeder (fix #5)."""

import os

import torch

from src.utils import set_seeds


def test_set_seeds_is_reproducible():
    set_seeds(123)
    a = torch.rand(5)
    set_seeds(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_set_seeds_sets_pythonhashseed():
    set_seeds(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_set_seeds_default_seed_is_42():
    set_seeds()
    assert os.environ["PYTHONHASHSEED"] == "42"
