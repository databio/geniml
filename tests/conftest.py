import pytest


def pytest_addoption(parser):
    """
    Adding options in commandline for pytest. The options decide which tests to skip
    """
    parser.addoption(
        "--bedbase",
        action="store_true",
        default=False,
        help="Run tests that needs connecting to BEDbase",
    )

    parser.addoption(
        "--qdrant",
        action="store_true",
        default=False,
        help="Run tests that needs connecting to Qdrant",
    )

    parser.addoption(
        "--r2vhf",
        action="store_true",
        default=False,
        help="Run tests that needs importing Region2Vec models form huggingface",
    )
