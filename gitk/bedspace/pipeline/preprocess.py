from .helpers import data_prepration


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
    :param labels: Labels string
    """
    # preprocess the data
    print("Running preprocess...")

    # do other stuff ...
    data_prep = data_prepration(input, metadata, universe, output, labels)

    # do mroe stuff ....
