from enmt.dataset import Split
from enmt.results import Dataset
from datasets import load_dataset


class OpenSubtitles(Dataset):
    def __init__(self, dataset_name="open_subtitles", lang1="en", lang2="sk", split=Split.EVAL, test_size=1.0, valid_size=40000, seed=1) -> None:
        # Dataset, self
        super().__init__(dataset_name, lang1=lang1, lang2=lang2, split=split, test_size=test_size, valid_size=valid_size, seed=seed)

    def load(self, dataset_name="open_subtitles", lang1="en", lang2="sk"):
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

            if self.valid_size != 0:
                validation = new['train'].train_test_split(test_size=self.valid_size, seed=self.seed)
                sets['train'] = validation['train']
                sets['val'] = validation['test']
            else:
                sets['train'] = new['train']

            sets['test'] = new['test']

        self.sets = sets
