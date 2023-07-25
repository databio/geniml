import os

from typing import Union, List, Tuple, Dict

from ..io import Region


# better name?
def extract_regions_from_bed_or_list(regions: Union[str, List[str], List[Region]]) -> List[Region]:
    """
    Validate the input for the regions. A universe accepts a lot of forms of input,
    so we need to check what the user has passed. It also standardizes the input to a list
    of tuples of chr, start, end.

    Users are allowed to pass a list of regions, a list of tuples with chr, start, and end, or
    a path to a BED file.

    :param regions: The regions to use for tokenization. This can be thought of as a vocabulary.
                    This can be either a list of regions or a path to a BED file containing regions.
    :return list: A list of tuples of chr, start, end.
    """
    # check for bedfile
    if isinstance(regions, str):
        # ensure that the file exists
        if not os.path.exists(regions):
            raise FileNotFoundError(f"Could not find file {regions} containing regions.")
        file = regions  # rename for clarity
        regions: List[Region] = []
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                chr, start, end = line.strip().split("\t")
                regions.append(Region(chr, int(start), int(end)))
        return regions

    # check for list of chr_start_end strings
    elif isinstance(regions, list) and all([isinstance(r, str) for r in regions]):
        regions_parsed: List[Region] = []
        for region in regions:
            chr, start, end = region.split("_")
            regions_parsed.append(Region(chr, int(start), int(end)))
        return regions_parsed

    # check for list of Region objects
    elif isinstance(regions, list) and all([isinstance(r, Region) for r in regions]):
        return regions
    else:
        raise ValueError(
            "The regions must be either a list of regions or a path to a BED file containing regions."
        )
