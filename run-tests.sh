#!/bin/bash

pip install . --user
python3 ./tests/pypackage_test.py
./tests/shell_test.sh
