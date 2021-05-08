# Experiments

## Repository Structure

File/folder description/contents

- *Analyze_Tokenizer.ipynb* - Just for debugging the BiLSTM tokenizer
- *code* - Contains the model + dataset loader code (main part of this project)
- *CrowdTangle_Bert.ipynb* - DistilBERT based model training code
- *CrowdTangle_BiLSTM.ipynb* - BiLSTM model training code
- *data* - Folder containing the preprocessed dataset
- *evaluation-bilstm.ipynb* - BiLSTM model evaluation code
- *evaluation-bert.ipynb* - DistilBERT based model evaluation code
- *figures* - Figures generated from the evaluation notebooks
- *PlainDataloaderTests.ipynb* - Debugging the dataloader
- *Simple Base Model.ipynb'* - LinearModel training + evaluation
- *start.sh* - Docker container start script


## Usage

If there's a `docker-compose.yml` you just need to have Docker installed and run
`docker-compose up` to get an up and running jupyter notebook.

Otherwise a `pipenv.lock` file is provided containing the dependencies.

## Experiments

There's a second `README.md` file inside the `crowdtangle` folder which explains the
notebooks used in the experiments.