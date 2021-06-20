# %%
import pytorch_lightning as pl
from nlp_utils.data_module import SemEvalDataModule
from nlp_utils.model import CustomDistilBertModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

seed_everything(42)

import os

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)

# %%
config = {
    "batch_size": 32,
    "dataset_path": "../../data/raw/SemEval/",
    "learning_rate": 0.001,
    "column_goldlabel": "Stance",
    "category_group_id": False,
    "category_type": False,
    "category_tld": False,
}

# %%
data_module = SemEvalDataModule(num_workers=4, config=config)
data_module.setup("")
# tokenizer for decoding sentences
tokenizer = data_module.get_tokenizer()
config["vocab_size"] = len(data_module.vocab)

# %%
model = CustomDistilBertModel(config)

early_stop_callback = EarlyStopping(
    monitor="val_epoch_F1",
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode="max",
    stopping_threshold=0.70,
)

cwd = os.getcwd()
save_folder = os.path.join(cwd, "../logs/StancePrediction_SemEval")


class MyPrintingCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_fit_end(self, trainer, pl_module):
        print("Finished training")

    def on_test_start(self, trainer, pl_module):
        print("Start to test")

    def on_test_end(self, trainer, pl_module):
        print("Finished testing")


checkpoint_callback = ModelCheckpoint(
    monitor="val_epoch_F1",
    filename="{epoch}-{val_loss:.2f}-{val_epoch_F1:.2f}",
    save_top_k=3,
    mode="max",
)

# %%
# Standard trainer
trainer = pl.Trainer(
    gpus=0,
    log_every_n_steps=1,
    flush_logs_every_n_steps=1,
    callbacks=[MyPrintingCallback(), checkpoint_callback],
    deterministic=True,
    default_root_dir=save_folder,
    max_epochs=1,
)  # gradient_clip_val=0.5, stochastic_weight_avg=True, check_val_every_n_epoch=10, num_sanity_val_steps=2, overfit_batches=0.01

# %% [markdown]
# ## Train

# %%
# train model
trainer.fit(model, datamodule=data_module)

trainer.test(model, datamodule=data_module)
