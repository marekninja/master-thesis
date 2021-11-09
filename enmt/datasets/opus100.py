from enmt.results import Dataset
from datasets import load_dataset


class Opus100(Dataset):
    def __init__(self, dataset_name="opus100", lang1="en", lang2="sk") -> None:
        # Dataset, self
        super().__init__(dataset_name, lang1, lang2)

    def load(self, dataset_name="opus100", lang1="en", lang2="sk"):
        return load_dataset(
            dataset_name, lang1+"-"+lang2)
