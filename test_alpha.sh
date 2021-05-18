#!/bin/bash
# Test to validate that python implementation matches R 

cd rr_python
python3 main_avg_deriv.py 

cd ../rrr_gas_blackbox
Rscript main_gas.R