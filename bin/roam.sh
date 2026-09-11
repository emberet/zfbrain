#!/bin/sh
# The fish. Sources .env (the single secrets/flags file) and execs roam.py in
# the venv. ZF_ALLOW_BROWSER=1 in .env is the gate that lets it open Chromium.
cd "$(dirname "$0")/.." || exit 1
set -a; [ -f .env ] && . ./.env; set +a
exec .venv/bin/python -u roam.py
