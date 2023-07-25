from ..utils import *
from typing import List

from intervaltree import Interval


class Region(Interval):
    def __new__(cls, chr: str, start: int, stop: int, data=None):
        return super(Region, cls).__new__(cls, start, stop, data)

    def __init__(self, chr: str, start: int, stop: int, data=None):
        # no need to call super().__init__() because namedtuple doesn't have __init__()
        self.chr = chr

    @property
    def start(self):
        return self.begin


# TODO: This belongs somewhere else
class RegionSet(object):
    def __init__(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                chr, start, stop = line.split("\t")
                self.regions.append(Region(chr, int(start), int(stop)))


# TODO: This belongs somewhere else; does it even make sense?
class TokenizedRegionSet(object):
    """Represents a tokenized region set"""

    def __init__(self, tokens: np.ndarray, universe: RegionSet):
        self.tokens = tokens
        self.universe = universe


# Write a class representing a collection of RegionSets
# TODO: This shouldn't read in the actual files, it should just represent the files and use lazy loading
class RegionSetCollection(object):
    """Represents a collection of RegionSets"""

    def __init__(self, region_sets: List[RegionSet] = None, file_globs: List[str] = None):
        if region_sets:
            self.region_sets = region_sets
        elif file_globs:
            self.region_sets = []
            for glob in file_globs:
                self.region_sets.extend([RegionSet(path) for path in glob.glob(glob)])

    def __getitem__(self, key):
        return self.region_sets[key]

    def __len__(self):
        return len(self.region_sets)


# Do we need an EmbeddingSet class?
class EmbeddingSet(object):
    """Represents embeddings and labels"""

    embeddings: np.ndarray
    labels: list
