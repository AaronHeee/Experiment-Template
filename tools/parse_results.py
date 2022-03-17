"""
    This script is used to print tabular results from ./experiments
    The printed results can be easily copied and pasted directly to google sheet (and notion)
"""

import json
import os
import sys

DIR = sys.argv[1]
METRICS = [] # TODO: e.g. ['ndcg', 'acc']
TARGET_FILE = "" # TODO: e.g. 'test.log'

print(f"{'model,': <70}" + ",".join(f"{m: >10}" for m in  METRICS))

for path in [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(DIR)) for f in fn]:
    if TARGET_FILE in path:
        try:
            test = json.load(open(path, 'r'))
            print(f"{path.replace(DIR, '').replace(TARGET_FILE, '')+',': <70}" + ','.join([f"{test[M]: >10.4f}" for M in METRICS if M in test]))
        except:
            pass