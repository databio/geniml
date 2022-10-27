import os
import yaml
import gensim
from csv import Sniffer

delim_sniffer = Sniffer()


def write_model_to_PEP(model_path: str, out_path: str):
    """
    Convert a gensim `.model` file to a PEP that is readable by RegionSet2Vec.

    :param str model_path: Path to the gensim `.model` file
    :param str out_path: Path to write out the new model
    """

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    model = gensim.models.Word2Vec.load(model_path)
    _meta = {}
    _meta["pep_version"] = "2.0.0"
    _meta["epochs"] = model.epochs
    _meta["embedding_dimensions"] = model.vector_size
    model_file = os.path.join(out_path, "model.yml")

    with open(model_file, "w+") as fh:
        yaml.dump(_meta, fh)

    for region in model.wv.vocab:
        region_vector = model.wv.get_vector(region)

    embedding_file = os.path.join(out_path, "embeddings.csv")

    with open(embedding_file, "w+") as fh:
        header = [
            "sample_name",
            "chr",
            "start",
            "end",
            *[f"dim{i+1}" for i in range(model.vector_size)],
        ]
        fh.write(",".join(header) + "\n")
        for i, region in enumerate(model.wv.vocab):
            # assume first represents all
            if i == 0:
                delim = delim_sniffer.sniff(region).delimiter
            region_vector = [str(v) for v in model.wv.get_vector(region).tolist()]
            sample_name = f"r{i}"
            chr, start, end = region.split(delim)

            values = [sample_name, chr, start, end, *region_vector]

            fh.write(",".join(values) + "\n")
