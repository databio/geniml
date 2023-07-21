import argparse
import os
import shlex
import subprocess

from gitk.region2vec import utils


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
            for path in glob.glob(glob)
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
            sort_process = subprocess.Popen(shlex.split(f"sort -k1,1V -k2,2n {input_path}"), stdout=temp_path)
            sort_process.communicate()
            # tokenize the sorted file
            bedtools_process = subprocess.Popen(
                shlex.split(f"{bedtools_path} intersect -a {universe} -b {temp_path} -u -f {fraction}"),
                stdout=output_file,
            )
            bedtools_process.communicate()



def bedtools_tokenization(
    f: str,
    bedtools_path: str,
    data_folder: str,
    target_folder: str,
    universe: str,
    fraction: float,
) -> None:
    """Uses bedtools to tokenize a raw BED file.

    Args:
        f (str): File name.
        bedtools_path (str): Path to a bedtools binary.
        data_folder (str): The folder where raw BED files reside.
        target_folder (str): The folder that stores tokenized BED files.
        universe (str): Path to a universe file.
        fraction (float): A parameter for bedtools.intersect.
    """
    fname = os.path.join(data_folder, f)
    temp = os.path.join(target_folder, f + "_sorted")
    target = os.path.join(target_folder, f)
    with open(temp, "w") as f_temp:
        subprocess.run(shlex.split(f"sort -k1,1V -k2,2n {fname}"), stdout=f_temp)
    with open(target, "w") as f_target:
        subprocess.run(
            shlex.split(f"{bedtools_path} intersect -a {universe} -b {temp} -u -f {fraction}"),
            stdout=f_target,
        )
    os.remove(temp)


def generate_tokens(
    raw_data_folder: str,
    token_folder: str,
    universe: str,
    file_list: str,
    bedtools: str,
    fraction: float,
) -> None:
    """Tokenizes raw BED files specified by file_list.

    Tokenizes raw BED files specified by file_list. First, checks existing files
    in token_folder. If token_folder has all the tokenized BED files, then does
    nothing. Otherwise, tokenizes raw BED files that are in file_list but not
    in token_folder.

    Args:
        raw_data_folder (str): The foder where raw BED files reside.
        token_folder (str): The folder to store tokenized BED files.
        universe (str): The path to a universe file.
        file_list (str): The path to a file which contains selected BED files per row.
        bedtools (str): The path to a bedtools binary.
        fraction (float): A parameter for bedtools.intersect.
    """
    usize = 0
    with open(universe, "r") as f:
        for _ in f:
            usize += 1
    print(f"\033[93mUniverse size is {usize}\033[00m")

    all_set = []
    with open(file_list, "r") as fin:
        for fname in fin:
            name = fname.strip()
            all_set.append(name)
    all_set = set(all_set)

    if os.path.exists(token_folder):
        files = os.listdir(token_folder)
        existing_set = set([f.strip() for f in files])
        not_covered = all_set - existing_set
        number = len(not_covered)
        if number == 0:
            print(f"Use the existing folder {token_folder}", flush=True)
            return
        else:
            print(
                f"Folder {token_folder} exists with {number} files not processed. Continue...",
                flush=True,
            )
    else:
        os.makedirs(token_folder)
        not_covered = all_set
    for f in not_covered:
        bedtools_tokenization(f, bedtools, raw_data_folder, token_folder, universe, fraction)


def main(args: argparse.Namespace):
    """Generates tokenized BED files.

    Calls generate_tokens using the arguments in args. Prints status
    information.

    Args:
        args (argparse.Namespace): See the definition of the ArgumentParser.
    """
    local_timer = utils.Timer()
    print(f"Entering hard tokenization. Results stored in {args.token_folder}")
    generate_tokens(
        args.data_folder,
        args.token_folder,
        args.universe,
        args.file_list,
        args.bedtools_path,
        args.fraction,
    )
    tokenization_time = local_timer.t()
    print(f"Hard tokenization takes {utils.time_str(tokenization_time)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-folder",
        type=str,
        default="/scratch/gz5hp/encode3/cell/datasets",
        help="path to the folder that stores BED files",
    )
    parser.add_argument(
        "--file-list",
        type=str,
        default="/home/gz5hp/encode3_proj/all_file_list.txt",
        help="list of BED files that will be tokenized",
    )

    parser.add_argument(
        "--token-folder",
        type=str,
        default="/scratch/gz5hp/encode3/cell/tokens",
        help="folder that stores tokenized files",
    )
    # parameters for hard tokenization
    parser.add_argument(
        "--universe",
        type=str,
        default="/home/gz5hp/encode3_proj/GRCh38-universe.bed",
        help="path to a universe file",
    )
    parser.add_argument(
        "--bedtools-path",
        type=str,
        default="/scratch/gz5hp/genomes/bedtools",
        help="path to the bedtools binary",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0e-9,
        help="a parameter for bedtools.intersect",
    )

    args = parser.parse_args()
    if os.path.exists(args.file_list):
        main(args)
