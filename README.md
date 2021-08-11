# Global Explainability for understanding opinions on social media

This project aims to explain classifiers that predict the potential impact and source of a social media post.
Additionally we classify stance and provide global explanations for it as well.

## Abstract

In this age of social media we have a immense
amount of data that these platforms can provide. It becomes extremely hard to find new
and innovative ways to extract value from that
data. In this paper we work on analyzing the
impact, ideology and stance of several social
media channels to enhance the understanding
of opinions on social media. For this we collected our own dataset in the form of Facebook
posts from CrowdTangle and also used the SemEval Stance dataset consisting of tweets with
annotated sentiment and stance. Additionally
we provide more insights into the model output by using several global eXplainable Artificial Intelligence (XAI) techniques. Specifically, we leverage SAGE and Neuron Attribution for global explainability.

## Structure

Code: consists of a standard config, the data modules and the models

Data: consists of the raw data for our prediction tasks

Logs: consists of Sage plots, clustered vocabularies and model parameters

_Important note, versioning of ray results, lightning_logs and predictions do match to ensure tensorboard logs and model checkpoints to be consistent. May lead to confusion because of missing version numbers for ray results or predictions._

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
