#!/usr/bin/env bash
docker run -v "$(pwd)"/:/home/jovyan --env NB_UID=1017 --user root -p 8888:8888 -p 6006:6006 --gpus all ct5
