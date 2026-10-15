"""Exercise the actual EventVLA write/read methods without starting a server."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'third_party/EventVLA/EventVLA'
sys.path[:0] = [str(CODE), str(CODE / 'examples/RoboTwin-Mem/eval_files')]
from model2robotwin_mem_interface import ModelClient


def client(capacity):
    value = ModelClient.__new__(ModelClient)
    value.keyframe_image_memory = []
    value.max_keyframe_images = capacity
    value.qwen_memory_injection_config = {}
    value.memory_config = {}
    value.temporal_view_names = ['cam_high']
    value.memory_injection_mode = 'pure_image_keyframe_memory'
    value.image_size = (3, 2)
    return value


def populate(value):
    for step in range(20):
        value._commit_keyframe_image_memory([np.full((2, 3, 3), step, np.uint8)], step, 0.8)


def test_hook_fifo_matches_original_memory_and_payload(monkeypatch):
    monkeypatch.delenv('RECOVERABILITY_MEMORY_POLICY', raising=False)
    original = client(4)
    populate(original)
    monkeypatch.setenv('RECOVERABILITY_MEMORY_POLICY', 'fifo')
    adapted = client(4)
    populate(adapted)
    original_images, original_metas, original_steps = original._build_keyframe_memory_images(20)
    images, metas, steps = adapted._build_keyframe_memory_images(20)
    assert steps == original_steps
    assert metas == original_metas
    assert all(np.array_equal(a, b) for a, b in zip(images, original_images))


@pytest.mark.parametrize('strategy', ['fifo', 'uniform'])
def test_zero_capacity_erases_stored_payload(monkeypatch, strategy):
    monkeypatch.setenv('RECOVERABILITY_MEMORY_POLICY', strategy)
    value = client(0)
    populate(value)
    assert value.keyframe_image_memory == []
    assert value._build_keyframe_memory_images(20) == ([], [], [])


def test_future_commits_are_not_read(monkeypatch):
    monkeypatch.setenv('RECOVERABILITY_MEMORY_POLICY', 'fifo')
    value = client(4)
    populate(value)
    images, metas, steps = value._build_keyframe_memory_images(17)
    assert steps == [16, 17]
    assert len(images) == len(metas) == 2
