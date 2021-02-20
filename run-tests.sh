#!/bin/bash

pip install .
python3 ./tests/pypackage_test.py
./tests/shell_test.sh
