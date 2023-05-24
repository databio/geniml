import os
import numpy as np
import random
import glob
import time
import datetime
import argparse
import os
from gitk.region2vec import utils
import pickle


class BEDDataset:
    def __init__(self, args, file_list):
        self.links = []
        self.args = args
        self.meta_data = dict()
        self.file2idx = dict()
        with open(file_list, "r") as f:
            for idx, line in enumerate(f):
                filename = line.strip()
                self.links.append(filename)
                self.file2idx[filename] = idx

        self.nfiles = len(self.links)

    def regions2sentences_sampling(self, src_path, dst_path):
        with open(dst_fname, "w") as fout:
            for fname in self.links:
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

    def regions2sentences(self, src_path, dst_path):
        with open(dst_path, "w") as f_out:
            for fname in self.links:
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
    def __init__(self, matrix):
        self.mat = [[] for i in range(len(matrix))]
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:
                    self.mat[i].append(j)

    def regions2sentences(self, dst_path):
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


def main(args):
    DATA_FOLDER = os.path.join(args.save_dir, "shuffled_datasets")
    os.makedirs(DATA_FOLDER, exist_ok=True)
    src_path = args.tokenization_folder
    worker_id = args.worker_id
    random.seed(worker_id)
    np.random.seed(worker_id)
    if args.data_type == "files":
        dataset = BEDDataset(args, args.file_list)
    else:
        with open(args.mat_path, "rb") as f:
            matrix = pickle.load(f)
        dataset = MatrixDataset(matrix)
    pool = args.pool
    utils.log(
        f"[{worker_id}] Creating shuffled datasets in \033[93m{DATA_FOLDER}\033[00m"
    )

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
    parser.add_argument("--file_list", help="path to a file list")
    parser.add_argument("--tokenization_mode", help="tokenization mode")
    parser.add_argument("--tokenization_folder", help="path to the tokenized regions")
    parser.add_argument(
        "--save_dir", help="parent folder to generated shuffled datasets"
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=3,
        help="maximum number of shuffled datasets before consuming one",
    )
    parser.add_argument(
        "--worker_id",
        type=int,
        default=0,
        help="maximum number of shuffled datasets before consuming one",
    )
    parser.add_argument(
        "--number", type=int, default=1000, help="total number of shuffled datasets"
    )

    args = parser.parse_args()

    main(args)
