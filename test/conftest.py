import pytest


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_epochs": 10,
        "device": "cpu",
    }
