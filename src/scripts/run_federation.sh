#!/usr/bin/env bash

set -euo pipefail

JSON_FILE="$1"

exec erl -noshell -hidden -s run_federation start \
     -extra "${JSON_FILE}"