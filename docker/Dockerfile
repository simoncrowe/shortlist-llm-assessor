FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl build-essential

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local' \
    POETRY_VERSION=1.8.3

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /code

COPY poetry.lock pyproject.toml .
RUN poetry install --no-interaction --no-ansi

COPY llm_assessor ./llm_assessor

ENTRYPOINT ["python3", "/code/llm_assessor"]
