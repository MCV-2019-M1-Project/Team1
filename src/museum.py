from pipeline_qsd1_w1_offline import run as run_offline_pipeline
from pipeline_qsd1_w1 import run as run_qsd1_pipeline
from pipeline_qsd2_w1 import run as run_qsd2_pipeline
import pickle
from pathlib import Path
import os

run_offline_pipeline()
result_qsd1 = run_qsd1_pipeline()
print(result_qsd1)
path = os.path.join(os.path.dirname(__file__), '../week1/QST1/method1')
with open('{}/result.pkl'.format(path), 'wb') as f:
    pickle.dump(result_qsd1, f)

result_qsd2 = run_qsd2_pipeline()
path = os.path.join(os.path.dirname(__file__), '../week1/QST2/method1')
with open('{}/result.pkl'.format(path), 'wb') as f:
    pickle.dump(result_qsd2, f)