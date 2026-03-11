import os
import shlex
import subprocess
import tempfile

from geniml.io import RegionSet

from . import FileTokenizer


class BEDToolsTokenizer(FileTokenizer):
    """A tokenizer that uses bedtools to tokenize BED files"""

    def __init__(self, bedtools_path: str, universe_path: str = None, fraction: float = 0.5):
        """Initialize a BEDToolsTokenizer

        Args:
            bedtools_path (str): Path to a bedtools binary.
            universe_path (str): Path to a universe BED file.
            fraction (float): Minimum overlap fraction. Defaults to 0.5.
        """
        self.bedtools_path = bedtools_path
        self.universe_path = universe_path
        self.fraction = fraction

    def tokenize(self, input_globs: list[str], universe_path: str = None) -> RegionSet:
        """Tokenize a RegionSet using bedtools"""

        universe_path = universe_path or self.universe_path

        # loop through globs and tokenize each file
        for glob_pattern in input_globs:
            import glob as glob_module

            for path in glob_module.glob(glob_pattern):
                self._tokenize_one(path, universe_path)

    def _tokenize_one(self, input_path: str, universe_path: str):
        output_path = os.path.join(input_path, "tokenized.bed")
        bedtools_path = self.bedtools_path
        universe = universe_path
        fraction = self.fraction
        # bedtools can't actually read from stdin, so we have to use a temporary file...

        # sort_process = subprocess.Popen(shlex.split(f"sort -k1,1V -k2,2n {input_path}"), stdout=subprocess.PIPE)
        # bedtools_process = subprocess.Popen(
        #     shlex.split(f"{bedtools_path} intersect -a {universe} -b  -u -f {fraction}"),
        #     stdin = sort_process.stdout,
        #     stdout = output_file,
        # )
        # bedtools_process.communicate()

        # get a temporary file path using tempfile

        with tempfile.NamedTemporaryFile() as temp_path, open(output_path, "w") as output_file:
            # sort the input file
            sort_process = subprocess.Popen(
                shlex.split(f"sort -k1,1V -k2,2n {input_path}"), stdout=temp_path
            )
            sort_process.communicate()
            # tokenize the sorted file
            bedtools_process = subprocess.Popen(
                shlex.split(
                    f"{bedtools_path} intersect -a {universe} -b {temp_path} -u -f {fraction}"
                ),
                stdout=output_file,
            )
            bedtools_process.communicate()
