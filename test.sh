#!/bin/bash
set -e

# LegalQA Task 2 Main Test Execution Gate
# Executes syntax, schema, notebooks, and full test suite

python scripts/verify_prepush.py "$@"
