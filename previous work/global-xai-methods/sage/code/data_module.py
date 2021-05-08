import pytorch_lightning as pl
from torchtext.experimental.datasets import IMDB
from torch.utils.data import DataLoader, Subset
import torch
from torch import nn

class Collator:
    
    def __init__(self, vocab):
        self.pad_idx = vocab['<pad>']
        
    def collate(self, batch):
        labels, text = zip(*batch)
        
        labels = torch.FloatTensor(labels)
        
        text = nn.utils.rnn.pad_sequence(text, padding_value = self.pad_idx)
        
        return labels, text

class IMDBDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.trainset = None
        self.valset = None
        self.testset = None
        self.collator = None

    def setup(self, stage):
        if self.trainset is None:
            self.setup_datasets()
    
    def setup_datasets(self):
        self.trainset, test_val_set = IMDB(data_select=('train', 'test'))
        self.valset = Subset(test_val_set, range(0, int(len(test_val_set)*0.1)))
        self.testset = Subset(test_val_set, range(int(len(test_val_set)*0.1) + 1, len(test_val_set)))
        self.collator = Collator(self.get_vocab())
    
    def train_dataloader(self):
        return DataLoader(
            self.trainset, 
            shuffle=True,
            batch_size=self.batch_size, 
            collate_fn=self.collator.collate, 
            num_workers=self.num_workers)
                                               
    def val_dataloader(self):
        return DataLoader(
            self.valset, 
            batch_size=self.batch_size, 
            collate_fn=self.collator.collate,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.testset, 
            batch_size=self.batch_size, 
            collate_fn=self.collator.collate,
            num_workers=self.num_workers)

    def get_vocab(self):
        return self.trainset.get_vocab()