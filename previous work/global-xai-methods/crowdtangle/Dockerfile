# Start from a core stack version
FROM jupyter/datascience-notebook
# Install in the default python3 environment
RUN pip install 'torch' && pip install 'transformers'
RUN pip install tqdm boto3 requests regex sentencepiece sacremoses tldextract
RUN pip install pytorch_lightning
RUN pip install torchsummary
RUN pip install torchtext
EXPOSE 8888
