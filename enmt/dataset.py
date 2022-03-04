from enum import Enum

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

import re
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class Split(Enum):
    """Split of dataset wanted.

    Args:
        Enum (TRAIN): For training
        Enum (VAL): For validation
        Enum (EVAL): For evaluation

    """
    TRAIN = "train"
    VAL = "validate"
    EVAL = "evaluation"

class Dataset():

    # FIXME maybe split is not needed
    def __init__(self, dataset_name:str, lang1:str, lang2:str, split: Split, test_size:float, valid_size=40000, seed=1) -> None:
        self.dataset = self.load(dataset_name, lang1, lang2)
        self.name = dataset_name
        self.source_lang = lang1
        self.target_lang = lang2
        self.test_size = test_size
        self.split = split
        self.sets = None
        self.seed = seed
        self.valid_size = valid_size

    def __getitem__(self, key):
        if self.sets is not None:
            return self.sets[key]
        else:
            return None

    def load(self, dataset_name, lang1, lang2):
        self.dataset = load_dataset(
            dataset_name, lang1=lang1, lang2=lang2)

    # TODO this should be dataset specific
    def _check_split(self, dataset):
        """

        Args:
            dataset: tokenized dataset; split into train and test
        Returns:
            modifies in-place

        """
        raise NotImplementedError(f"Need to define dataset split handling! Dataset is: {self.name}")
        # example code:
        # keys = dataset.keys()
        #
        # sets = {}
        # for i in ['train', 'validation', 'test']:
        #     if i in keys:
        #         sets[i] = dataset[i]
        #     else:
        #         sets[i] = None
        #
        # if 'train' in keys and 'test' not in keys:
        #     if 'train' in keys:
        #         new = dataset['train'].train_test_split(test_size=0.2, seed=1)
        #         sets['train'] = new['train']
        #         sets['test'] = new['test']
        #
        # if 'train' not in keys:
        #     raise RuntimeError("Dataset does not have 'train' split")
        #
        # self.sets = sets

    def preprocess(self, tokenizer, max_input_length=512, max_target_length=512, prefix=""):



        # preprocessed_path = os.path.join(dir_path+"/datasets/preprocessed_pickled/"+self.name+"_"+tokenizer.name_or_path+"_"\
        #                     +tokenizer.source_lang+"-"+tokenizer.target_lang+"_preprocessed.dump")

        # RX = re.compile('([\\`/<>:"\|\?\*])')

        # dump_name = self.name + "_" + tokenizer.name_or_path + "_"+ tokenizer.source_lang + "-" +\
        #             tokenizer.target_lang + "_preprocessed_dump"
        dump_name = self.name + "_" + tokenizer.alias + "_" + tokenizer.source_lang + "-" + \
                    tokenizer.target_lang + "_preprocessed_dump"
        dump_name = re.sub(r'[\\`/<>:"|?*]', '-', dump_name)
        preprocessed_path = os.path.join(dir_path,"datasets","preprocessed_pickled", dump_name)


        tokenized_dataset = None
        if os.path.isdir(preprocessed_path):
            print("Found preprocessed dataset...")

            tokenized_dataset = DatasetDict.load_from_disk(preprocessed_path)
            # with open(preprocessed_path, 'rb') as pickle_file:
            #     tokenized_dataset = pickle.load(pickle_file)
        else:
            print("Preprocessed dataset not found. Preprocessing in progress...")

            self.tokenizer = tokenizer
            self.max_input_length = max_input_length
            self.max_target_length = max_target_length
            self.prefix = prefix
            tokenized_dataset = self.dataset.map(self._preprocess_function, batched=True)
            tokenized_dataset.save_to_disk(preprocessed_path)

            # print(dir_path)
            # print(os.getcwd())
            # with open(preprocessed_path, 'wb') as pickle_file:
            #     pickle.dumps(tokenized_dataset, pickle_file)

        self._check_split(tokenized_dataset)



    def _preprocess_function(self, examples):
        inputs = [self.prefix + ex[self.source_lang]
                  for ex in examples["translation"]]
        targets = [ex[self.target_lang] for ex in examples["translation"]]
        model_inputs = self.tokenizer(
            inputs, max_length=self.max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets, max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
