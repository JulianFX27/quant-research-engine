#!/usr/bin/env bash
set -euo pipefail

python -m pip install -q pytest
pytest -q

