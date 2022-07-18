#!/usr/bin/env python
# coding: utf-8

import sys
import os
from utils import Model    

year = int(sys.argv[1])
month = int(sys.argv[2])

categorical = ['PUlocationID', 'DOlocationID']
os.environ['OUTPUT_FILE_PATTERN'] = f'taxi_type=fhv_year={year:04d}_month={month:02d}.parquet'


if __name__ == '__main__':
    model = Model('model.bin', year, month).predict_nsave(categorical)


