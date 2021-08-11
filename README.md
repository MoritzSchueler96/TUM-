# Global Explainability for understanding opinions on social media

This project aims to explain classifiers that predict the potential impact and source of a social media post.
Additionally we classify stance and provide global explanations for it as well.

## Structure

Code: consists of a standard config, the data modules and the models

Data: consists of the raw data for our prediction tasks

Logs: consists of Sage plots, clustered vocabularies and model parameters

Important note, versioning of ray results and lightning_logs do match to ensure tensorboard logs and model checkpoints to be consistent. May lead to confusion because of missing version numbers for ray results

Presentation: consists of the final presentation of the project

Report: consists of the final report of the project

Previous work: consists of the repository of former students working on a similar task

Tasks: Notebooks for training and analyzing the models designed for stance & impact prediction

## Installation

1. Create a conda environment with Python 3.7

    ```python
    conda create -n your-env-name python=3.7
    ```

2. Activate your environment

    ```python
    conda activate your-env-name
    ```

3. Install the requirements

    ```python
    pip install -r requirements.txt
    ```

4. Install utils

    Go into /code folder and install nlp_utils package

    ```python
    cd code/

    pip install -e .
    ```
