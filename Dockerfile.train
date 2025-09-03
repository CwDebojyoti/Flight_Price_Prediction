FROM python:3.10-slim

WORKDIR /app
COPY app/ .
COPY requirements.txt ./requirements.txt

ENV PORT=8080

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3","main.py"]