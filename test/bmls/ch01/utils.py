__author__ = 'nenggong'

import os

DATA_DIR = os.path.join(os.path.relpath(__file__), "data")

CHART_DIR = os.path.join(os.path.realpath(__file__), "charts")

for d in [DATA_DIR, CHART_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)