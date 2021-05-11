import pytorch_lightning as pl
import os
from torch.utils.data import DataLoader, Subset, Dataset
import torch
from torch import nn
from transformers import AutoTokenizer, DistilBertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from nlp_utils.config import create_config
import tldextract

tld_extractor = tldextract.TLDExtract()
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence


class GroupId2Name:
    """
    Helper function to transform facebook group IDs to group names.
    """

    def __init__(self, config):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, config["dataset_path"])
        df = pd.read_csv(filename, low_memory=False)
        self.group_id2name = {
            x[1]: x[0]
            for x in df.drop_duplicates(subset=["Name", "Facebook Id"])[
                ["Name", "Facebook Id"]
            ].values.tolist()
        }

    def __call__(self, groupId):
        return self.group_id2name[groupId]


def inverse_transform(vector, config=None, encoder=None):
    """
    returns the actual categories from a one-hot-encoding.
    Based on the config, some categories might be missing (e.g. group_id for rtnews)
    """
    config = create_config(config)
    result = {}
    decoded = list(encoder.inverse_transform(vector)[0])

    if config["category_group_id"]:
        result["group_id"] = decoded.pop(0)

    if config["category_type"]:
        result["post_type"] = decoded.pop(0)

    if config["category_tld"]:
        result["domain"] = decoded.pop(0)

    return result


def get_classes_per_row(dataset, config):
    """
    takes a batch or full dataset and returns a tuple of all classes which are enabled in the config object
    """
    for x in dataset:
        result = []
        if config["category_group_id"]:
            result.append(x["Facebook Id"])

        if config["category_type"]:
            result.append(x["Type"])

        if config["category_tld"]:
            domain = ""

            if not pd.isna(x["Link"]):
                tld_result = tld_extractor(x["Link"])
                domain = tld_result.domain

            result.append(domain)

        yield tuple(result)


class Collator:
    """
    helper class to transform a batch so it can be fed into the DistilBERT based model
    """

    def __init__(self, tokenizer, class_encoder, config):
        self.config = create_config(config)
        self.tokenizer = tokenizer
        self.class_encoder = class_encoder

    def collate(self, batch):
        labels, features = zip(*batch)

        encoded_texts = self.tokenizer(
            [row["Text"] for row in features],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded_classes = self.class_encoder.transform(
            list(get_classes_per_row(features, self.config))
        )

        return (
            torch.FloatTensor(labels),
            encoded_texts,
            torch.FloatTensor(encoded_classes),
            features,
        )


class CrowdTangleDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        return self.labels[index], self.texts[index]

    def __len__(self):
        return len(self.texts)


def create_datasets(class_encoder, config={}):
    """
    creates a train, validation and test dataset.
    The trainset will be shuffeled. 42 is used as random seed to get deterministic results.
    """
    config = create_config(config)

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, config["dataset_path"])
    df = pd.read_csv(filename, low_memory=False)

    df["Text"] = (
        df["Message"].fillna("").astype(str)
        + " "
        + df["Image Text"].fillna("").astype(str)
        + " "
        + df["Link Text"].fillna("").astype(str)
        + " "
        + df["Description"].fillna("").astype(str)
    )

    raw_records = df.to_dict("records")
    raw_labels = df[config["column_goldlabel"]].to_list()

    # Split train / val_test date
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        raw_records, raw_labels, test_size=0.4, random_state=42
    )

    # initialize encoder using training data
    class_encoder.fit(list(get_classes_per_row(X_train, config)))

    # Split validation / test set
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=42
    )

    return (
        CrowdTangleDataset(X_train, y_train),
        CrowdTangleDataset(X_val, y_val),
        CrowdTangleDataset(X_test, y_test),
    )


class CrowdTangleDataModule(pl.LightningDataModule):
    """
    Data module for the DistilBERT based model + linear model
    """

    def __init__(self, num_workers=4, config={}):
        super().__init__()
        config = create_config(config)
        self.batch_size = config["batch_size"]
        self.num_workers = num_workers

        self.collator = None
        self.trainset = None
        self.valset = None
        self.testset = None
        self.vocab = []
        self.config = config
        # self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.class_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    def setup(self, stage):
        if self.trainset is None:
            self.collator = Collator(self.tokenizer, self.class_encoder, self.config)
            self.trainset, self.valset, self.testset = create_datasets(
                self.class_encoder, self.config
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def get_tokenizer(self):
        return self.tokenizer


class PlainCollator:
    """
        helper class to transform a batch so it can be fed into the BiLSTM model
        """

    def __init__(self, vocab, tokenizer, class_encoder, config):
        self.config = create_config(config)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.class_encoder = class_encoder

    def collate(self, batch):
        labels, features = zip(*batch)

        encoded_texts = []
        for row in features:
            encoded_texts.append(
                torch.tensor(
                    [self.vocab[token] for token in self.tokenizer(row["Text"])]
                )
            )

        encoded_classes = self.class_encoder.transform(
            list(get_classes_per_row(features, self.config))
        )

        return (
            torch.FloatTensor(labels),
            pad_sequence(
                encoded_texts, batch_first=True, padding_value=self.vocab["<pad>"]
            ),
            torch.FloatTensor(encoded_classes),
            features,
        )


class PlainCrowdTangleDataModule(pl.LightningDataModule):
    """
        Data module for the BiLSTM model
    """

    def __init__(self, num_workers=4, config={}):
        super().__init__()
        config = create_config(config)
        self.batch_size = config["batch_size"]
        self.num_workers = num_workers

        self.collator = None
        self.trainset = None
        self.valset = None
        self.testset = None
        self.vocab = None
        self.config = config
        self.tokenizer = get_tokenizer("basic_english")
        self.class_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    def setup(self, stage):
        if self.trainset is None:
            self.trainset, self.valset, self.testset = create_datasets(
                self.class_encoder, self.config
            )
            self._build_vocab()
            self.collator = PlainCollator(
                self.vocab, self.tokenizer, self.class_encoder, self.config
            )

    def _build_vocab(self):
        counter = Counter()
        for (label, features) in iter(self.trainset):
            counter.update(self.tokenizer(features["Text"]))
        self.vocab = Vocab(counter, min_freq=self.config["vocab_min_freq"])

    def _decode_vocab_vec(self):
        a = 1

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def get_tokenizer(self):
        return self.tokenizer


class SemEvalDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        return self.labels[index], self.texts[index]

    def __len__(self):
        return len(self.texts)


def create_datasets(class_encoder, config={}):
    """
    creates a train, validation and test dataset.
    The trainset will be shuffeled. 42 is used as random seed to get deterministic results.
    """
    config = create_config(config)

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, config["dataset_path"])
    df = pd.read_csv(filename, low_memory=False)

    raw_records = df.to_dict("records")
    raw_labels = df[config["column_goldlabel"]].to_list()

    # Split train / val_test date
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        raw_records, raw_labels, test_size=0.4, random_state=42
    )

    # initialize encoder using training data
    class_encoder.fit(list(get_classes_per_row(X_train, config)))

    # Split validation / test set
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=42
    )

    return (
        SemEvalDataset(X_train, y_train),
        SemEvalDataset(X_val, y_val),
        SemEvalDataset(X_test, y_test),
    )


class SemEvalDataModule(pl.LightningDataModule):
    """
    Data module for the DistilBERT based model + linear model
    """

    def __init__(self, num_workers=4, config={}):
        super().__init__()
        config = create_config(config)
        self.batch_size = config["batch_size"]
        self.num_workers = num_workers

        self.collator = None
        self.trainset = None
        self.valset = None
        self.testset = None
        self.vocab = []
        self.config = config
        # self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.class_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    def setup(self, stage):
        if self.trainset is None:
            self.collator = Collator(self.tokenizer, self.class_encoder, self.config)
            self.trainset, self.valset, self.testset = create_datasets(
                self.class_encoder, self.config
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def get_tokenizer(self):
        return self.tokenizer
