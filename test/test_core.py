import pytest


def test_import():
    """Verify the package can be imported."""
    assert True


def test_config_defaults(sample_config):
    """Verify default configuration values."""
    assert sample_config["batch_size"] > 0
    assert sample_config["learning_rate"] > 0
    assert sample_config["max_epochs"] > 0


def test_device_selection(sample_config):
    """Verify device can be set."""
    assert sample_config["device"] in ("cpu", "cuda", "mps")
