# Dockerfile for ParaCite
FROM python:3.10-slim

WORKDIR /app

COPY . /app

# If you add requirements.txt later, uncomment below:
# RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3"]
