import pandas as pd
import utils_fastai
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
import util

from fastai.vision.all import *
from fastai.data.all import *

M = pd.read_parquet('master.parquet')



print('set seed')
seed_pt = int(input())

print("folder to start with (e.g. 5, for all folders, Max for just Max)")
start_folder = input()
if start_folder.isnumeric():
    start_folder = int(start_folder)
    folders = [str(item) for item in range(start_folder,365,5)]+['Max']
else:
    folders = [start_folder]
print('folders are :', folders)    

grayscale = util.get_bool('grayscale?(e.g. True/False)')

categories = ['A', 'B','C', 'D', 'E', 'F', 'G', 'J', 'P']
print("remove any groups? (e.g. P,PF,etc)")
remove_groups = input()
new_categories = tuple(sorted(set(categories) - set(remove_groups)))
pollock_groups = list(set(['P','J'])-set(remove_groups))
print(pollock_groups)
      
valid_list = pd.read_csv(os.path.join('Paintings','Processed','Raw','valid_lists','gcp_resnet50_72_Full_10-01-2022','valid_list')).valid_list.tolist()
valid_list = [item for item in valid_list if not item.startswith(tuple(set(remove_groups)))]

M = M[M.file.str.startswith(new_categories)]

print('starting model run')

metrics = utils_fastai.model_run(M, 'gcp_resnet50_valid',
                  Descreened = False,
                  valid_set = valid_list,
                  seed = seed_pt,
                  thread_num = False,
                  computer = 'GCP',
                  px_size = 256,
                  grayscale = grayscale,
                  epochs = 1,
                  model = resnet50,
                  folders = folders,
                  remove_duplicates = True,
                  notes = 'gcp run',
                  pollock_groups = pollock_groups,
                  multipath = Path('Paintings/Processed/Descreened/')
                   )
print('finished model run')
print(metrics.to_string())
