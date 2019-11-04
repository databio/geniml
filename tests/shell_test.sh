#!/bin/bash

bedshift --bedfile test.bed --droprate 0.1 --addrate 0.2 --addmean 320.0 --addstdev 30.0 --shiftrate 0.3 --shiftmean 0.0  --shiftstdev 150.0  --cutrate 0.1  --mergerate 0.2  --outputfile sh_output.bed
