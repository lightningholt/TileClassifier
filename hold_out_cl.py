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
      
valid_list = ['A12', 'A22(R)', 'A27', 'A31(X)', 'A33(X)', 'A35(X)', 'A47', 'A48',
       'A50(X)', 'A56', 'A60(R)', 'A9(right)', 'C14', 'C2', 'C21', 'C22',
       'C29', 'C35', 'C45', 'C51', 'C53', 'C63', 'C64', 'C66', 'D15',
       'D16', 'D17', 'D22', 'D26', 'D4', 'E14', 'E3', 'E7', 'F102',
       'F105', 'F108', 'F13', 'F16', 'F22', 'F23', 'F30', 'F34', 'F39',
       'F45', 'F47', 'F49', 'F54', 'F60', 'F65', 'F70', 'F72', 'F79',
       'F80', 'F81', 'F89', 'G16', 'G17', 'G18', 'G19', 'G2', 'G31',
       'G46', 'G52', 'G53', 'G54', 'G60', 'G61', 'G7', 'G70', 'G72',
       'G79', 'G96', 'G99', 'JPCR_00093', 'JPCR_00173', 'JPCR_00174',
       'JPCR_00182', 'JPCR_00188', 'JPCR_00190', 'JPCR_00193',
       'JPCR_00195', 'JPCR_00209', 'JPCR_00210', 'JPCR_00221(Right)',
       'JPCR_00224', 'JPCR_00228', 'JPCR_00236', 'JPCR_00237',
       'JPCR_00239', 'JPCR_00240', 'JPCR_00244', 'JPCR_00249',
       'JPCR_00259', 'JPCR_00266', 'JPCR_00273', 'JPCR_00275',
       'JPCR_00276', 'JPCR_00277', 'JPCR_00278', 'JPCR_00284',
       'JPCR_00287', 'JPCR_00289', 'JPCR_00298', 'JPCR_00364',
       'JPCR_00370', 'JPCR_00372', 'JPCR_01028', 'P100(O)', 'P110(L)',
       'P112(W)', 'P118(B)', 'P2(V)', 'P21(W)', 'P37(W)', 'P38(O Left)',
       'P38(O Right)', 'P48(V)', 'P54(R)', 'P55(V)', 'P57(V)',
       'P58(S Left)', 'P58(S Right)', 'P62(F)', 'P63(S)', 'P67(F)',
       'P68(V)', 'P83(P)', 'P84(O)', 'P87(S)', 'P9(W)']
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
                  hold_out_catalog= ['194', '198', '201', '208', '1031', '1030',
                                     '1034', '231', '251','280', '285', '286', '318', 
                                     '792', '801', '808', '802', '818', '825', '824',
                                     '820', '823', '815', '816', '826', '360', '855',
                                     '797', '1088b', '1019', '162', '164', '380',
                                     '?', '346', '1033', '130', '206', '218', '225',
                                     '262', '288', '291', '292', '293',
                                     '294', '295', '296', '299', '300', '310', '311',
                                     '312', '316','317', '366', '1002', '1032',
                                     '1038', '1088'],
                  multipath = Path('Paintings/Processed/Descreened/')
                   )
print('finished model run')
print(metrics.to_string())
