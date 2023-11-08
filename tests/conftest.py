import pytest


def pytest_addoption(parser):
    """
    Adding options in commandline for pytest. The options decide which tests to skip
    To actually run some test, use this command in terminal:
    pytest <test file name> <option>
    For example:
    pytest test_text2bednn.py --qdrant --r2vhf
    pytest --bedbase
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
    # add an --all option to set other option as True
