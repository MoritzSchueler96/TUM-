FROM python:3.8

RUN apt-get update && apt-get install -y cmake

WORKDIR /usr/src/app
RUN useradd -m python && chown -R python .

COPY Pipfile .
RUN pip install --no-cache-dir pipenv
RUN pip install --no-cache-dir torch
#RUN pip install --pre torch torchtext -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html && \
#    pip install pytorch-lightning jupyter sage-importance
# add a non root user for script execution

# install script dependencies
USER python
RUN pipenv --site-packages install
COPY . .

# install latest version of torchtext
WORKDIR /home/python
RUN git clone https://github.com/pytorch/text.git
WORKDIR /home/python/text
RUN git submodule update --init --recursive &&\
    python setup.py bdist_wheel &&\
    pip install dist/*.whl

WORKDIR /usr/src/app

# start the notebook server
#CMD [ "pipenv" "run" "jupyter notebook" ]
ENTRYPOINT ["./bin/entrypoint.sh"]
