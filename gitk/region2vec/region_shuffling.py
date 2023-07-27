import argparse
import datetime
import glob
import os
import pickle
import random
import time
from typing import List

import numpy as np

from gitk.region2vec import utils


class BEDDataset:
    """Wraps a set of BED files in a BEDDataset object.

    Stores the information of a set of BED files.
    Generates a new dataset with regions shuffled in BED files.

    Attributes:
        filename_list: A list of BED file names.
        nfiles: The number of BED files in the BEDdataset object.
    """

    def __init__(self, file_list: str) -> None:
        """Initializes a BEDDataset object.

        Args:
            file_list (str): A file storing a list of BED file names that
                should be included in the dataset.
        """
        self.filename_list = []
        with open(file_list, "r") as f:
            for idx, line in enumerate(f):
                filename = line.strip()
                self.filename_list.append(filename)

        self.nfiles = len(self.filename_list)

    def regions2sentences_sampling(self, src_path: str, dst_path: str) -> None:
        """Constructs a sentence by sampling regions from a BED file.

        Each region in a BED file has a probability. Constructs a sentence by
        sampling regions in a BED file based on their probabilities.

        Args:
            src_path (str): The folder where BED files reside.
            dst_path (str): The destination file that stores all the generated
                BED files; each line has regions sampled from a BED file.
        """
        with open(dst_fname, "w") as fout:
            for fname in self.filename_list:
                src_fname = os.path.join(src_path, fname)
                sentence = []
                probs = []
                with open(src_fname, "r") as f:
                    for line in f:
                        elements = line.strip().split("\t")
                        word = elements[0].strip()
                        sentence.append(word)
                        probs.append(float(elements[-2].strip()))
                probs = np.array(probs)
                probs = probs / probs.sum()
                sentence = np.array(sentence)

                sampled_sentence = np.random.choice(sentence, len(probs), p=probs)
                # sampled_sentence = list(set(sampled_sentence))
                sampled_sentence = sampled_sentence.tolist()
                str_sent = " ".join(sampled_sentence)

                fout.write(str_sent)
                fout.write("\n")

    def regions2sentences(self, src_path: str, dst_path: str) -> None:
        """Concatenates all regions in a BED file randomly into a sentence.

        This functions is called in the hard tokenization mode.

        Args:
            src_path (str): The folder where BED files reside.
            dst_path (str): The destination file that stores all the generated
                BED files; each line has all the regions from a BED file.
        """
        with open(dst_path, "w") as f_out:
            for fname in self.filename_list:
                src_fname = os.path.join(src_path, fname)
                sentence = []
                with open(src_fname, "r") as f:
                    for line in f:
                        elements = line.strip().split("\t")[0:3]
                        chr_name = elements[0].strip()
                        start = elements[1].strip()
                        end = elements[2].strip()
                        word = chr_name + ":" + start + "-" + end
                        sentence.append(word)
                random.shuffle(sentence)  # shuffle the regions in the sentence
                str_sent = " ".join(sentence)
                f_out.write(str_sent)
                f_out.write("\n")


class MatrixDataset:
    """Wraps the binary representation of BED files into a MatrixDataset.

    Stores the information of a set of BED files.
    Generates a new dataset with regions shuffled in BED files.
    """

    def __init__(self, matrix: List[List[int]]):
        """Initializes a MatrixDataset object with matrix.

        Args:
            matrix (list[list[int]]): The binary representation of BED files.
                Each row represents a BED file. Each column denotes a region.
                Each element denotes the presence (1) or absence (0) of a
                region in a BED file.
        """
        self.mat = [[] for i in range(len(matrix))]
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:
                    self.mat[i].append(j)

    def regions2sentences(self, dst_path: str) -> None:
        """Concatenates all regions in a BED file randomly into a sentence.

        This functions is called in the hard tokenization mode.

        Args:
            dst_path (str): The destination file that stores all the generated
                BED files; each line has all the regions from a BED file.
        """
        with open(dst_path, "w") as f_out:
            for i in range(len(self.mat)):
                sentence = []
                for j in range(len(self.mat[i])):
                    sentence.append(self.mat[i][j])
                if len(sentence) > 0:
                    random.shuffle(sentence)  # shuffle the regions in the sentence
                    str_sent = " ".join(sentence)
                    f_out.write(str_sent)
                    f_out.write("\n")


def main(args: argparse.Namespace) -> None:
    """Generates shuffled datasets.

    Called internally as a subprocess by region2vec in main.py.

    Args:
        args (argparse.Namespace): See the definition of ArgumentParser.
    """
    DATA_FOLDER = os.path.join(args.save_dir, "shuffled_datasets")
    os.makedirs(DATA_FOLDER, exist_ok=True)
    src_path = args.tokenization_folder
    worker_id = args.worker_id
    random.seed(worker_id)
    np.random.seed(worker_id)
    if args.data_type == "files":
        dataset = BEDDataset(args.file_list)
    else:
        with open(args.mat_path, "rb") as f:
            matrix = pickle.load(f)
        dataset = MatrixDataset(matrix)
    pool = args.pool
    utils.log(f"[{worker_id}] Creating shuffled datasets in \033[93m{DATA_FOLDER}\033[00m")

    for i in range(pool):
        name_used = os.path.join(DATA_FOLDER, f"pool{worker_id}-{i}used")
        name_using = os.path.join(DATA_FOLDER, f"pool{worker_id}-{i}using")
        name_creating = os.path.join(DATA_FOLDER, f"pool{worker_id}-{i}creating")
        name = os.path.join(DATA_FOLDER, f"pool{worker_id}-{i}")
        if os.path.exists(name_using):
            print("File exists")
            return
        if os.path.exists(name_used):
            print("File exists")
            return
        if os.path.exists(name):
            print("File exists")
            return
        if os.path.exists(name_creating):
            print("File exists")
            return
        # create an empty file
        with open(name_used, "w") as f:
            pass

    num_created = 0
    while True:
        if num_created == args.number:
            break
        # determine whether to create a new dataset
        files = glob.glob(os.path.join(DATA_FOLDER, f"pool{worker_id}*used"))
        if len(files) == 0:
            time.sleep(1)  # wait for 10 seconds
            # print('Waiting for the data to be consumed',end="\r")
        else:
            # delete the used dataset and generate a new dataset in the same foler
            sel_file = files[random.randint(0, len(files) - 1)]
            fname = sel_file.split("/")[-1][:-4]
            os.system(f"rm -f {sel_file}")  # delete the dataset
            dpath = os.path.join(DATA_FOLDER, fname + "creating")
            with open(dpath, "w") as f:
                pass
            if args.tokenization_mode == "hard":
                dataset.regions2sentences(src_path, dpath)
            else:
                dataset.regions2sentences_sampling(src_path, dpath)

            num_created += 1
            # print('[',datetime.datetime.now(),']',' Created %dth dataset' % num_created)
            dst_name = os.path.join(DATA_FOLDER, fname)
            os.rename(dpath, dst_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentence Generation")
    parser.add_argument("--file-list", help="path to a file list")
    parser.add_argument("--tokenization-mode", help="tokenization mode")
    parser.add_argument(
        "--tokenization-folder",
        help="path to the folder that saves tokenized regions",
    )
    parser.add_argument("--save-dir", help="parent folder to generated shuffled datasets")
    parser.add_argument(
        "--pool",
        type=int,
        default=3,
        help="maximum number of shuffled datasets before consuming one",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="used in the parallel mode",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=1000,
        help="number of shuffling the whole dataset",
    )

    args = parser.parse_args()

    main(args)
