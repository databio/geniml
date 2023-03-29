import argparse
import os
from gitk.tokenization import utils
import subprocess
import shlex


def bedtool_tokenization(
    f, bedtool_path, data_folder, target_folder, universe, fraction
):
    fname = os.path.join(data_folder, f)
    temp = os.path.join(target_folder, f + "_sorted")
    target = os.path.join(target_folder, f)
    with open(temp, "w") as f_temp:
        subprocess.run(shlex.split("sort -k1,1 -k2,2n {}".format(fname)), stdout=f_temp)
    with open(target, "w") as f_target:
        subprocess.run(
            shlex.split(
                "{} intersect -a {} -b {} -u -f {} ".format(
                    bedtool_path, universe, temp, fraction
                )
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
    print("\033[93mUniverse size is {}\033[00m".format(usize))

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
            print("Use the existing name {}".format(token_folder), flush=True)
            return 0
        else:
            print(
                "Folder {} exists with {} files not processed. Continue...".format(
                    token_folder, number
                ),
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
    print("Entering hard tokenization. Results stored in {}".format(args.token_folder))
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
    print("Hard tokenization takes {}".format(utils.time_str(tokenization_time)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder", type=str, default="/scratch/gz5hp/encode3/cell/datasets"
    )
    parser.add_argument(
        "--file_list", type=str, default="/home/gz5hp/encode3_proj/all_file_list.txt"
    )

    parser.add_argument(
        "--token_folder", type=str, default="/scratch/gz5hp/encode3/cell/tokens"
    )
    # parameters for hard tokenization
    parser.add_argument(
        "--universe", type=str, default="/home/gz5hp/encode3_proj/GRCh38-universe.bed"
    )
    parser.add_argument(
        "--bedtools_path", type=str, default="/scratch/gz5hp/genomes/bedtools"
    )
    parser.add_argument("--fraction", type=float, default=1.0e-9)

    args = parser.parse_args()
    if os.path.exists(args.file_list):
        main(args)
