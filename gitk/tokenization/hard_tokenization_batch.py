import argparse
import os
import shlex
import subprocess

from gitk.tokenization import utils


def bedtool_tokenization(
    f, bedtool_path, data_folder, target_folder, universe, fraction
):
    fname = os.path.join(data_folder, f)
    temp = os.path.join(target_folder, f + "_sorted")
    target = os.path.join(target_folder, f)
    with open(temp, "w") as f_temp:
        subprocess.run(shlex.split(f"sort -k1,1V -k2,2n {fname}"), stdout=f_temp)
    with open(target, "w") as f_target:
        subprocess.run(
            shlex.split(
                f"{bedtool_path} intersect -a {universe} -b {temp} -u -f {fraction} "
            ),
            stdout=f_target,
        )
    os.remove(temp)


def generate_tokens(
    raw_data_folder, token_folder, universe, file_list, bedtool, fraction
):
    """
    Perform hard tokenization on the bed files
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
            return 0
        else:
            print(
                f"Folder {token_folder} exists with {number} files not processed. Continue...",
                flush=True,
            )
    else:
        os.makedirs(token_folder)
        not_covered = all_set
    for f in not_covered:
        bedtool_tokenization(
            f, bedtool, raw_data_folder, token_folder, universe, fraction
        )
    return 0


def main(args):
    local_timer = utils.Timer()
    print(f"Entering hard tokenization. Results stored in {args.token_folder}")
    status = generate_tokens(
        args.data_folder,
        args.token_folder,
        args.universe,
        args.file_list,
        args.bedtools_path,
        args.fraction,
    )
    if status < 0:
        return
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
