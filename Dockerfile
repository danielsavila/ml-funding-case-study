# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION} AS base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir --upgrade -r requirements.txt

COPY app.py ./
COPY cleaned_data.py ./
COPY creating_data.py ./
COPY rfmodel.pkl ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]