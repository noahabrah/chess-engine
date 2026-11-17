import pytest

from engine import nnue


@pytest.fixture(autouse=True)
def pesto_by_default(monkeypatch):
    """Tests use PeSTO by default, so they don't depend on whichever network was trained last.

    The NNUE tests switch the network on themselves (see test_nnue.py).
    """
    monkeypatch.setattr(nnue, "network", None)
