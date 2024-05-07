import pandas as pd
import utils_fastai
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL

from fastai.vision.all import *
from fastai.data.all import *


M = pd.read_parquet('master.parquet')
# removing max from folders
folders = ['10','100','105', '110', '115', '120', '125',
         '130', '135', '140', '145', '15', '150', '155', '160',
         '165', '170', '175', '180', '185', '190', '195', '20',
         '200', '205', '210', '215', '220', '225', '230', '235',
         '240', '245', '25', '250', '255', '260', '265', '270',
         '275', '280', '285', '290', '295', '30', '300', '305',
         '310', '315', '320', '325', '330', '335', '340', '345',
         '35', '350', '355', '360', '40', '45', '5', '50',
         '55', '60', '65', '70', '75', '80', '85', '90',
         '95', 'Max']
print('starting model run')

print('set seed')
seed_pt = int(input())

metrics = utils_fastai.model_run(M, 'gcp_resnet50',
                  Descreened = False,
                  valid_set = 0.2,
                  seed = seed_pt,
                  thread_num = False,
                  computer = 'GCP',
                  px_size = 256,
                  grayscale = False,
                  epochs = 1,
                  model = resnet50,
                  folders = folders,
                  remove_duplicates = True,
                  notes = 'gcp run'
                   )
print('finished model run')
print(metrics.to_string())
