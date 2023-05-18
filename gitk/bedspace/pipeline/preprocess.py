import logging

from ..const import PKG_NAME

_LOGGER = logging.getLogger(PKG_NAME)


def main(
    input: str,
    metadata: str,
    universe: str,
    output: str,
    labels: str,
):
    """
    Main function for the preprocess pipeline

    :param input: Path to input bed files
    :param metadata: Path to metadata file
    :param universe: Path to universe file
    :param output: Path to output folder
    :param labels: Labels string (cell_type,target)
    """
    _LOGGER.info("Running preprocess...")

    # PLACE CODE FOR RUNNING PREPROCESS HERE
