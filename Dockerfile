# Cambia solo questa riga!
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app

ENV GIT_PYTHON_REFRESH=quiet

CMD ["python", "-m", "src.data.make_datasets"]