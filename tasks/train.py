# %%
import pytorch_lightning as pl
from nlp_utils.data_module import SemEvalDataModule
from nlp_utils.model import CustomDistilBertModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

import random
import re
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
    # might not be called due to scheduler and reporter which cancel training early if results don't look promising
    trainer.test(model, datamodule=data_module)


config = {
    "dataset_path": "../../data/raw/SemEval/",
    "learning_rate": tune.sample_from(lambda: abs(random.gauss(1e-3, 1e-3))),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "epochs": 20,
    "num_trials": 50,
}


callbacks = [MyPrintingCallback(), checkpoint_callback, callback]

scheduler = ASHAScheduler(max_t=config["epochs"], grace_period=1, reduction_factor=2)

reporter = CLIReporter(
    parameter_columns=["lr", "batch_size"],
    metric_columns=["loss", "mean_accuracy", "training_iteration"],
)

# ray.init(local_mode=True, num_cpus=4, num_gpus=0)  # for debugging

# get version
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split("(\d+)", text)]


log_dir = "../logs/StancePrediction_SemEval/lightning_logs/"
log_path = os.path.join(path, log_dir)
os.makedirs(os.path.dirname(log_path), exist_ok=True)
ver = os.listdir(os.path.join(path, log_dir))
ver.sort(key=natural_keys)
if ver:
    version = int(ver[-1].split("_", 2)[-1]) + 1
else:
    version = 0

analysis = tune.run(
    tune.with_parameters(
        train_tune, callbacks=callbacks, epochs=config["epochs"], gpus=0
    ),
    config=config,
    num_samples=config["num_trials"],
    local_dir=os.path.join(path, "../logs/StancePrediction_SemEval/ray_results"),
    name="version_" + str(version),
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
# df2 = analysis.dataframe(metric="val_epoch_F1", mode="max") # check how to include multiple metrics


print("Best hyperparameters found were: ", analysis.best_config)

# save dataframe with results from hyperparameter search as csv?
"""
# do some SHAP stuff :D
import shap
import scipy as sp
import torch

# SHAP Tokenization Function, without attention mask
def f_highest_prob_logit(x):
    tv = torch.tensor(
        [
            tokenizer_auto.encode(
                v, padding="max_length", max_length=sample_max_length, truncation=True
            )
            for v in x
        ]
    )  # .cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    score_most_prob = [max(x) for x in scores]
    # print(score_most_prob)
    return sp.special.logit(score_most_prob)


explainer = shap.Explainer(f_highest_prob_logit, tokenizer_auto)
shap_values = explainer(ag_news_few_samples["text"], fixed_context=1)

# true labels and predictions for reference
print(ag_news_few_samples["label"])
print(list(class_pred))
shap.plots.text(shap_values[10])

shap.plots.waterfall(shap_values[0])


# visualize the first prediction's explanation with a force plot
shap.initjs()
shap.plots.force(shap_values[0])


shap.initjs()
# visualize all the training set predictions
shap.plots.force(explainer.expected_value, shap_values.values)


shap.initjs()
# create a dependence scatter plot to show the effect of a single feature across the whole dataset
shap.plots.scatter(shap_values[:, "RM"], color=shap_values)


shap.initjs()
# summarize the effects of all the features
shap.plots.beeswarm(shap_values)

"""
