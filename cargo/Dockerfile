FROM python:3.8.11-slim as python-base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/cargo" \
    VENV_PATH="/cargo/.venv"
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

FROM python-base as builder-base
ENV POETRY_VERSION=1.0.0
RUN apt update && apt install -y --no-install-recommends curl build-essential
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
WORKDIR $PYSETUP_PATH
COPY ./pyproject.toml ./
RUN poetry install --no-dev

FROM python-base as production
ENV FASTAPI_ENV=production
COPY --from=builder-base $VENV_PATH $VENV_PATH
COPY ./cargo.py $PYSETUP_PATH/cargo.py
WORKDIR $PYSETUP_PATH
