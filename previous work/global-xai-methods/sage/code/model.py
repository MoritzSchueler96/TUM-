import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchtext.experimental.datasets import IMDB


class RNN(pl.LightningModule):
    """
    heavily inspired by
    https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
    """
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0)).squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        y, x = batch
        y, x = y.to(self.device), x.to(self.device)
        y_hat = self(x)
        loss = F.mse_loss(y, y_hat)
        
        return {'loss': loss, 'log': {'loss': loss}}
    
    def validation_step(self, batch, batch_idx):
        y, x = batch
        y, x = y.to(self.device), x.to(self.device)
        
        y_hat = self(x)
        
        loss =  F.mse_loss(y, y_hat)
        correct = torch.sum(y == y_hat)
        acc = correct / len(y)
        
        return {'val_loss': loss, 'val_acc': acc , 'log': {'val_loss': loss, 'val_acc': acc}}
        
        
    def validation_end(self, outputs):
        pass
   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    