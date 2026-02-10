FROM tensorflow/tensorflow:2.16.1-gpu

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app

RUN python -m venv --system-site-packages "${VIRTUAL_ENV}"

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && pip install -r /app/requirements.txt

COPY pyproject.toml /app/pyproject.toml
COPY studybuddy_ml /app/studybuddy_ml
COPY configs /app/configs

RUN pip install /app

CMD ["studybuddy-ml", "--help"]
