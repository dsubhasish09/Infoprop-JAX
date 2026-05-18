FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG PROJECT_DIR
ARG WANDB_KEY

RUN [ ! -z "${PROJECT_DIR}" ]
RUN [ ! -z "${WANDB_KEY}" ]

ENV PROJECT_PATH=${PROJECT_DIR}

RUN apt update && apt upgrade -y
RUN apt install -y git wget \
    && apt-get clean  
   

WORKDIR ${PROJECT_DIR}

# RUN curl -LsSf https://astral.sh/uv/install.sh | sh 
# RUN uv venv --python 3.10
# RUN source .venv/bin/activate
# RUN uv pip install -r requirements/requirements.txt
# RUN uv pip install -e .


