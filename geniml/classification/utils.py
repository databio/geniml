import scanpy as sc
from torch.utils.data import Dataset


class SingleCellClassificationDataset(Dataset):
    def __init__(self, data: sc.AnnData, label_key: str):
        """
        Initialize the dataset.

        :param sc.AnnData data: The data to use for training.
        :param str label_key: The key in the obs to use for the labels.
        :param float train_test_split: The fraction of the data to use for training.
        """
        self.data = data
        self.label_key = label_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :], self.data.obs[self.label_key].iloc[idx]
