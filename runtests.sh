#!/bin/bash

set -eux

uv run pytest -k test_linear
uv run pytest -k test_embedding
