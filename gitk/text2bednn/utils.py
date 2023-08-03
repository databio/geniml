import os
from typing import List, Tuple
import random


def data_split(full_list: List,
                    train_p=0.85*0.9,
                    valid_p=0.85*0.1,
                    seed_index: int = 10) -> Tuple[List, List, List]:
    """
    With a given folder of data, this function split the files
    into training set, validating set, and testing set.

    :param bed_folder: folder where bed files are stored
    :param train_p: proportion of files for training set
    :param valid_p: proportion of files for validating set
    and files will be copied to them.
    :param seed_index: index for random seed
    :return: lists of file names for training set, validating set, and training set.
    """

    # split training data and testing data
    # file_list = os.listdir(bed_folder)
    train_size = int(len(full_list) * train_p)
    validate_size = int(len(full_list) * valid_p)
    random.seed(seed_index)
    train_list = random.sample(full_list, train_size)
    validate_list = random.sample([content for content in full_list if content not in train_list], validate_size)
    test_list = [content for content in full_list if (content not in train_list and content not in validate_list)]

    # train_list.sort()
    # validate_list.sort()
    # test_list.sort()

    return train_list, validate_list, test_list


def metadata_line_process(metadata_line: str) -> str:
    """
    remove formatting characters and interval set name from metadata

    :param set_name: name of interval set
    :param metadata_line: the metadata text
    :return: the metadata text without interval set name and formatting characters
    """

    metadata_line = metadata_line.replace("\t", " ")
    metadata_line = metadata_line.replace("\n", "")

    return metadata_line


"""
ls | shuf -n 20 | xargs -I '{}' cp '{}' /home/claudehu/Desktop/repo/geniml_dev/tests/data/hg38_sample

for file in /home/claudehu/Desktop/repo/geniml_dev/tests/data/hg38_sample/*.bed; do
    key=$(basename "$file" | cut -d. -f1)
    grep -w "$key" experimentList_sorted_hg38.tab >> /home/claudehu/Desktop/repo/geniml_dev/tests/data/hg38_metadata_sample.tab
done

shuf -n 20 experimentList_sorted_hg38.tab >> /home/claudehu/Desktop/repo/geniml_dev/tests/data/hg38_metadata_sample.tab

sort -k1,1 -t$'\t' hg38_metadata_sample.tab > hg38_metadata_sample_sorted.tab


"""
