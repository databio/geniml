#!/bin/bash

bedshift --bedfile $(dirname "$0")/test.bed --droprate 0.1 --addrate 0.2 --addmean 320.0 --addstdev 30.0 --shiftrate 0.3 --shiftmean 0.0  --shiftstdev 150.0  --cutrate 0.1  --mergerate 0.2  --outputfile $(dirname "$0")/sh_output.bed

bedshift --bedfile $(dirname "$0")/test.bed --droprate 0.1 --addrate 0.2 --addfile $(dirname "$0")/test.bed --outputfile $(dirname "$0")/sh_output2.bed

bedshift --bedfile $(dirname "$0")/test.bed --addrate 0.1 --cutrate 0.5 --addfile $(dirname "$0")/test.bed --outputfile $(dirname "$0")/sh_output2.bed -r 10
