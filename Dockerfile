# quantcortex application image (Python 3.11).
FROM python:3.11-slim-bookworm@sha256:e2d3af735aff6eeee600b1933bedd99da6645fedf572cc12ef4cc1331f2ceebe

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    XDG_STATE_HOME=/tmp/state

WORKDIR /app

# Runtime library needed by LightGBM/XGBoost when optional models are mounted.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the locked operational stack first for deterministic layer caching.
COPY requirements/runtime.lock requirements/runtime.lock
RUN python -m pip install --no-cache-dir -r requirements/runtime.lock

RUN groupadd --gid 10001 quantcortex \
    && useradd --uid 10001 --gid quantcortex --create-home quantcortex

COPY --chown=quantcortex:quantcortex . .
USER quantcortex

CMD ["python", "-c", "import quantcortex.alpha, quantcortex.backtest, quantcortex.data, quantcortex.execution, quantcortex.portfolio, quantcortex.strategies; print('quantcortex container ready')"]
