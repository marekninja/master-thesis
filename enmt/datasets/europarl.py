from enmt.dataset import Split
from enmt.results import Dataset
from datasets import load_dataset



class EuroParl(Dataset):
    # FIXME maybe split is not needed
    def __init__(self, dataset_name="europarl_bilingual", lang1="en", lang2="sk", split=Split.EVAL, test_size=1.0, seed=1) -> None:
        # Dataset, self
        super().__init__(dataset_name, lang1=lang1, lang2=lang2, split=split, test_size=test_size, seed=seed)

    def load(self, dataset_name="europarl_bilingual", lang1="en", lang2="sk"):
        return load_dataset(
            dataset_name, lang1=lang1, lang2=lang2)

    def _check_split(self, dataset):
        keys = dataset.keys()

        sets = {}

        if 'train' not in keys:
            raise RuntimeError("Dataset does not have 'train' split")

        if self.test_size == 1.0:
            sets['test'] = dataset['train']
            del dataset['train']
        elif self.test_size > 0.0:
            new = dataset['train'].train_test_split(test_size=self.test_size, seed=self.seed)
            sets['train'] = new['train']
            sets['test'] = new['test']

        self.sets = sets
