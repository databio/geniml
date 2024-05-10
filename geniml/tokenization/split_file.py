import argparse
import os


def get_file_rows(file_path: str) -> int:
    """Counts how many files are included.

    Args:
        file_path (str): The path to a file.

    Returns:
        int: The number of rows in the file.
    """
    count = 0
    with open(file_path, "r") as f:
        for line in f:
            count += 1
    return count


def split_file(file_path: str, dest_folder: str, num_parts: int) -> None:
    """Splits a list of files into num_parts non-overlapping batches.

    This is a helper function for tokenization in parallel. The dest_folder
    must be empty.

    Args:
        file_path (str): The path to a file with BED file names per row.
        dest_folder (str): The folder to store split file lists.
        num_parts (int): Number of batches to split.
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
            fname = os.path.join(dest_folder, f"split_{index}.txt")
            with open(fname, "w") as fout:
                for _ in range(num_arr[index]):
                    fout.write(list_arr[pos])
                    fout.write("\n")
                    pos = pos + 1
        assert count == pos, "Missing some files"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", default="", help="path to a file list")
    parser.add_argument("--dest-folder", default="", help="where to store the split files")
    parser.add_argument(
        "--num-parts",
        type=int,
        default=5,
        help="split the original file list to the specified parts",
    )
    args = parser.parse_args()

    split_file(args.file_path, args.dest_folder, args.num_parts)
