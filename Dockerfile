FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libffi-dev build-essential

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "dramatiq", "jobs", "--processes=1", "--threads=1"]
