from . import FileTokenizer


class BEDToolsTokenizer(FileTokenizer):
    """A tokenizer that uses bedtools to tokenize BED files"""

    def __init__(self, bedtools_path: str, universe_path: str = None):
        """Initialize a BEDToolsTokenizer

        Args:
            bedtools_path (str): Path to a bedtools binary.
            universe_path (str): Path to a universe BED file.
        """
        self.bedtools_path = bedtools_path
        self.universe_path = universe_path

    def tokenize(self, input_globs: list[str], universe_path: str = None) -> RegionSet:
        """Tokenize a RegionSet using bedtools"""

        universe_path = universe_path or self.universe_path

        # loop through globs and tokenize each file
        for glob in input_globs:
            for path in glob.glob(glob):
                _tokenize_one(path, universe_path)

    def _tokenize_one(self, input_path: str, universe_path: str):
        output_path = os.path.join(input_path, "tokenized.bed")
        bedtools_path = self.bedtools_path
        # bedtools can't actually read from stdin, so we have to use a temporary file...

        # sort_process = subprocess.Popen(shlex.split(f"sort -k1,1V -k2,2n {input_path}"), stdout=subprocess.PIPE)
        # bedtools_process = subprocess.Popen(
        #     shlex.split(f"{bedtools_path} intersect -a {universe} -b  -u -f {fraction}"),
        #     stdin = sort_process.stdout,
        #     stdout = output_file,
        # )
        # bedtools_process.communicate()

        # get a temporary file path using tempfile
        import templfile

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
