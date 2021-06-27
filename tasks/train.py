# %%
import pytorch_lightning as pl
from nlp_utils.data_module import SemEvalDataModule
from nlp_utils.model import CustomDistilBertModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

import random

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)

seed_everything(42)

import os

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)

# %%

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.05,
    patience=3,
    verbose=False,
    mode="min",
    divergence_threshold=3.00,
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
    monitor="val_epoch_stance_F1",
    filename="{epoch}-{val_loss:.2f}-{val_epoch_stance_F1:.2f}",
    save_top_k=3,
    mode="max",
)

callback = TuneReportCallback(
    {"loss": "val_loss", "mean_F1": "val_epoch_F1"}, on="validation_end"
)


def train_tune(config, callbacks, epochs=10, gpus=0):
    data_module = SemEvalDataModule(num_workers=4, config=config)
    data_module.setup("")
    config["vocab_size"] = len(data_module.vocab)
    config["target_encoding"] = data_module.target_encoding
    config["stance_encoding"] = data_module.stance_encoding
    model = CustomDistilBertModel(config)
    trainer = pl.Trainer(
        gpus=gpus,
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
        callbacks=callbacks,
        deterministic=True,
        default_root_dir=save_folder,
        max_epochs=epochs,
    )  # gradient_clip_val=0.5, stochastic_weight_avg=True, check_val_every_n_epoch=10, num_sanity_val_steps=2, overfit_batches=0.01
    # logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


config = {
    "dataset_path": "../../data/raw/SemEval/",
    "learning_rate": tune.sample_from(lambda: abs(random.gauss(1e-3, 1e-3))),
    "batch_size": tune.choice([32, 64, 128]),
    "epochs": 1,
    "num_trials": 1,
}


callbacks = [MyPrintingCallback(), checkpoint_callback, callback]

scheduler = ASHAScheduler(max_t=config["epochs"], grace_period=1, reduction_factor=2)

reporter = CLIReporter(
    parameter_columns=["lr", "batch_size"],
    metric_columns=["loss", "mean_accuracy", "training_iteration"],
)

ray.init(local_mode=True, num_cpus=4, num_gpus=0)  # for debugging

analysis = tune.run(
    tune.with_parameters(
        train_tune, callbacks=callbacks, epochs=config["epochs"], gpus=0
    ),
    config=config,
    num_samples=config["num_trials"],
    local_dir=os.path.join(path, "../logs/StancePrediction_SemEval/ray_results"),
    name="version_0",
    metric="loss",
    mode="min",
    scheduler=scheduler,
    progress_reporter=reporter,
)
# metric="loss", mode="min", scheduler=scheduler, progress_reporter=reporter


best_trial = analysis.best_trial  # Get best trial
best_config = analysis.best_config  # Get best trial's hyperparameters
best_logdir = analysis.best_logdir  # Get best trial's logdir
best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
best_result = analysis.best_result  # Get best trial's last results
best_result_df = analysis.best_result_df  # Get best result as pandas dataframe

# Get a dataframe with the last results for each trial
df_results = analysis.results_df

# Get a dataframe of results for a specific score or mode
df = analysis.dataframe(metric="loss", mode="min")
df2 = analysis.dataframe(metric="val_epoch_F1", mode="max")


print("Best hyperparameters found were: ", analysis.best_config)
