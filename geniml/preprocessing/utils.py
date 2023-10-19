from tqdm.rich import tqdm

from ..io import Region

from ..utils import wordify_region


def make_vocab_from_bed(bed_file: str, vocab_file: str, n_unused: int = 10000):
    """
    Create a vocabulary file from a bed file. This includes
    the special tokens [PAD], [UNK], [CLS], [SEP], and [MASK] as
    well as unused tokens for new regions that may be encountered
    during training or pre-training.
    """
    with open(bed_file, "r") as f:
        regions = [line.strip().split("\t") for line in f.readlines()]

    with open(vocab_file, "w") as f:
        f.write("[PAD]\n")
        f.write("[UNK]\n")
        f.write("[CLS]\n")
        f.write("[SEP]\n")
        f.write("[MASK]\n")
        for region in tqdm(regions, desc="Writing regions"):
            r = Region(region[0], int(region[1]), int(region[2]))
            f.write(wordify_region(r) + "\n")

        # add unused tokens
        for i in tqdm(range(n_unused), desc="Writing unused tokens"):
            f.write(f"[UNUSED{i}]\n")
