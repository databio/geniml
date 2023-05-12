import os
import argparse


def get_file_rows(file_path):
    """
    Count how many files are included
    """
    count = 0
    with open(file_path, "r") as f:
        for line in f:
            count += 1
    return count


def split_file(file_path, dest_folder, num_parts):
    """
    Split a list of files into a specified non-overlapping batches

    file_path: path to a list of files
    dest_folder: folder to store file splits
    num_parts: number of parts needed
    """
    if os.path.exists(dest_folder):
        print("Folder exists")
        return
    os.makedirs(dest_folder)
    count = get_file_rows(file_path)

    list_arr = []
    with open(file_path, "r") as f:
        for line in f:
            list_arr.append(line.strip())

    num_per_file = int(count / num_parts)
    if num_per_file == 0:
        print("No need to split")
        return
    else:
        last_file_num = count - num_per_file * (num_parts - 1)
        num_arr = [num_per_file] * (num_parts - 1) + [last_file_num]
        pos = 0
        for index in range(num_parts):
            fname = os.path.join(dest_folder, "split_{}.txt".format(index))
            with open(fname, "w") as fout:
                for _ in range(num_arr[index]):
                    fout.write(list_arr[pos])
                    fout.write("\n")
                    pos = pos + 1
        assert count == pos, "Missing some files"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default="")
    parser.add_argument("--dest_folder", default="")
    parser.add_argument("--num_parts", type=int, default=5)
    args = parser.parse_args()

    split_file(args.file_path, args.dest_folder, args.num_parts)
