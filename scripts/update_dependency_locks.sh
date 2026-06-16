#!/usr/bin/env bash
set -euo pipefail

poetry lock
poetry export --only main --without-hashes --output requirements/core.lock
poetry export --only main,test --without-hashes --output requirements/test.lock
poetry export --only main,dev --without-hashes --output requirements/notebooks.lock
poetry export --only main,test,dev --without-hashes --output requirements/dev.lock
poetry export --only main -E brokers --without-hashes --output requirements/brokers.lock
poetry export --only main -E brokers -E providers -E storage -E regime \
  --without-hashes --output requirements/runtime.lock
poetry export --only build --without-hashes --output requirements/build.lock
poetry check
