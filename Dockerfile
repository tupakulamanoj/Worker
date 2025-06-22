FROM python:3.11-slim

WORKDIR /app

COPY . /app/

RUN pip install --upgrade pip
RUN pip install --upgrade -r requirements.txt

CMD ["python", "-m", "dramatiq", "jobs"]
