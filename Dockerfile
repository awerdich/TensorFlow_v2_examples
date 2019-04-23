FROM tensorflow/tensorflow:2.0.0a0-gpu-py3-jupyter AS base

#If you need a CPU-only version, us this:
#FROM tensorflow/tensorflow:2.0.0a0-py3-jupyter AS base

ARG DEV_tensorflow_v2_examples
ARG CI_USER_TOKEN
ARG CURRENT_UID
RUN echo "machine github.com\n  login $CI_USER_TOKEN\n" >~/.netrc

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_SRC=/src \
    PIPENV_HIDE_EMOJIS=true \
    PIPENV_COLORBLIND=true \
    PIPENV_NOSPIN=true

RUN apt-get update
RUN apt-get install git -y
RUN apt-get install python-gdcm -y
RUN apt-get install tcl-dev tk-dev python3-tk -y 
RUN apt-get install x11-xserver-utils -y
RUN apt-get install -qqy x11-apps -y

RUN pip install pipenv

WORKDIR /tf

COPY Pipfile .
COPY Pipfile.lock .

COPY setup.py .
COPY src/tensorflow_v2_examples/__init__.py src/tensorflow_v2_examples/__init__.py
COPY matplotlibrc ./.config/matplotlib/matplotlibrc

RUN pipenv install --system --deploy --ignore-pipfile --dev