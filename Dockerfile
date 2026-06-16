# quantcortex application image (Python 3.11).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for scientific wheels + psycopg2 (TimescaleDB).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libgomp1 libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip poetry-core \
    && pip install \
        numpy pandas scipy scikit-learn matplotlib pyarrow \
        redis sqlalchemy psycopg2-binary \
        pytest pytest-cov

# Copy the source tree.
COPY . .

CMD ["python", "-c", "import quantcortex.alpha, quantcortex.backtest, quantcortex.data, quantcortex.execution, quantcortex.portfolio, quantcortex.strategies; print('quantcortex container ready')"]
