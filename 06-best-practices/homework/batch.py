#!/usr/bin/env python
# coding: utf-8

import sys
from utils import Model    

year = int(sys.argv[1])
month = int(sys.argv[2])

categorical = ['PUlocationID', 'DOlocationID']

if __name__ == '__main__':
    model = Model('model.bin', year, month).predict_nsave(categorical)


