FROM python:3.13-slim

WORKDIR /app

# Install deps in their own layer so `docker build` doesn't re-download torch/transformers
# (the bulk of the image) on every source change.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

# ingestion/fixtures.py reads its stand-in corpus from tests/fixtures/ (see that module's
# docstring — it's a deliberate stand-in for real BigQuery ingestion, not test-only data)
COPY tests/fixtures ./tests/fixtures

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
