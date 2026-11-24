from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def cpu_reproducibility():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    yield
    torch.set_num_threads(previous)
