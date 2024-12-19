FROM python:3.9
RUN mkdir -p /app
RUN mkdir -p /tmp
WORKDIR /app
COPY pyproject.toml /app/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install --upgrade pip
RUN python -m pip install poetry
RUN python --version
RUN poetry --version
RUN poetry config virtualenvs.create false

RUN poetry install
#CMD [ "gunicorn",  "--worker-class", "gevent", "--workers", "2", "--bind", "0.0.0.0:8888", "app:app"]
CMD ["python", "-u", "100poseEstimationRun.py"]