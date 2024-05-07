#get_all_folder_preds.py

import pandas as pd
import utils_fastai
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
import util
from ast import literal_eval

from fastai.vision.all import *
from fastai.data.all import *

import rubric
import plotting



#set params
learner_folder = 'gcp_classifier-3_10-Max_color_10-15-2022'
sub_folder = 'preds'
raw_path = 'Paintings/Processed/Raw/'
descreened_path = 'Paintings/Processed/Descreened/'
extra_imitations_path = 'Extra_Imitations/Processed/Raw/'
extra_imitations_master_path = 'master_new_imits.parquet'
save_pre_path = 'runs'
cutoff = 0.7

#for viz
container = 'runs'
viz = 'viz'


# load in master
M=pd.read_parquet('master.parquet')



# load in metrics
metrics = pd.read_csv(Path(save_pre_path,learner_folder,'metrics.csv'))

hold_out_files_full =  utils_fastai.get_hold_out_from_metrics(M,metrics)

folders = metrics.folders.apply(eval).iloc[0]

print('your learner_folder is set to ' + learner_folder)
auto_decide = util.get_bool('automatically decide correct preds?(e.g. True/False)')
if auto_decide:
    assert not metrics.Descreened.iloc[0], 'this autodecider is not set up for descreened runs'
    multipath = metrics.multipath.iloc[0]
    if multipath:
        valid_artifacts_Raw = True 
        HO_replacement = True
        HO_artifacts_Raw = True

        valid_artifacts_Descreened = False
        HO_raw = False
        HO_artifacts_Descreened = False
    else:
        valid_artifacts_Descreened = True
        HO_raw = True
        HO_artifacts_Descreened = True

        valid_artifacts_Raw = False 
        HO_replacement = False
        HO_artifacts_Raw = False
    if os.path.exists(extra_imitations_path):
        extra_imitations = True
    else:
        print('extra imitations folder does not exist')
        extra_imitations = False
    print('Settings:')
    print('HO_replacement',HO_replacement)
    print('HO_raw',HO_raw)    
    print('valid_artifacts_Raw',valid_artifacts_Raw) 
    print('valid_artifacts_Descreened',valid_artifacts_Descreened)
    print('HO_artifacts_Raw',HO_artifacts_Raw)
    print('HO_artifacts_Descreened',HO_artifacts_Descreened)
    print('extra_imitations',extra_imitations)
else:
    HO_replacement = util.get_bool('do HO_replacement?(e.g. True/False)')
    HO_raw = util.get_bool('do HO_raw?(e.g. True/False)')
    valid_artifacts_Raw = util.get_bool('do valid_artifacts_raw?(e.g. True/False)')
    valid_artifacts_Descreened = util.get_bool('do valid_artifacts_descreened?(e.g. True/False)')
    HO_artifacts_Raw = util.get_bool('do HO_artifacts_raw?(e.g. True/False)')
    HO_artifacts_Descreened = util.get_bool('do HO_artifacts_descreened?(e.g. True/False)')
    extra_imitations = util.get_bool('do extra_imitations?(e.g. True/False)')
make_summary = util.get_bool('make summary?(e.g. True/False)')
make_viz = util.get_bool('make visualizations for problem paintings?(e.g. True/False)')



    

if HO_replacement:
    #do hold_out set (dynamic)
    save_folder = os.path.join(learner_folder,sub_folder,'HO_replacement')
    utils_fastai.get_learn_preds_wrapper(M[M.file.isin(hold_out_files_full)], learner_folder, save_folder,img_path = False,overwrite=True,save_pre_path = 'runs',folders=folders)
    
if HO_raw:
    #do hold_out set (dynamic)
    save_folder = os.path.join(learner_folder,sub_folder,'HO_raw')
    utils_fastai.get_learn_preds_wrapper(M[M.file.isin(hold_out_files_full)], learner_folder, save_folder,img_path = 'Paintings/Processed/Raw/',overwrite=True,save_pre_path = 'runs',folders=folders)

if valid_artifacts_Raw:
    #do Raw run of valid files with artifacts
    save_folder = os.path.join(learner_folder,sub_folder,'valid_artifacts_Raw')
    valid_set = literal_eval(metrics.valid_set.iloc[0])
    M_filtered = M[np.logical_and(M.file.isin(valid_set),M.artifacts == 'True')]
    utils_fastai.get_learn_preds_wrapper(M_filtered, learner_folder, save_folder,img_path = raw_path,overwrite=True,save_pre_path = save_pre_path,folders=folders)

if HO_artifacts_Raw:
    #do Raw run of hold_out files with artifacts
    save_folder = os.path.join(learner_folder,sub_folder,'HO_artifacts_Raw')
    valid_set = literal_eval(metrics.valid_set.iloc[0])
    M_filtered = M[np.logical_and(M.file.isin(hold_out_files_full),M.artifacts == 'True')]
    utils_fastai.get_learn_preds_wrapper(M_filtered, learner_folder, save_folder,img_path =raw_path,overwrite=True,save_pre_path = save_pre_path,folders=folders)
    
if valid_artifacts_Descreened:
    #do Raw run of valid files with artifacts
    save_folder = os.path.join(learner_folder,sub_folder,'valid_artifacts_Descreened')
    valid_set = literal_eval(metrics.valid_set.iloc[0])
    M_filtered = M[np.logical_and(M.file.isin(valid_set),M.artifacts == 'True')]
    utils_fastai.get_learn_preds_wrapper(M_filtered, learner_folder, save_folder,img_path = descreened_path,overwrite=True,save_pre_path = save_pre_path,folders=folders)

if HO_artifacts_Descreened:
    #do Raw run of hold_out files with artifacts
    save_folder = os.path.join(learner_folder,sub_folder,'HO_artifacts_Descreened')
    valid_set = literal_eval(metrics.valid_set.iloc[0])
    M_filtered = M[np.logical_and(M.file.isin(hold_out_files_full),M.artifacts == 'True')]
    utils_fastai.get_learn_preds_wrapper(M_filtered, learner_folder, save_folder,img_path =descreened_path,overwrite=True,save_pre_path = save_pre_path,folders=folders)

if extra_imitations:
    M_extra = pd.read_parquet(extra_imitations_master_path)
    #do run of extra_imitations
    save_folder = os.path.join(learner_folder,sub_folder,'extra_imitations')
    valid_set = literal_eval(metrics.valid_set.iloc[0])
    utils_fastai.get_learn_preds_wrapper(M_extra, learner_folder, save_folder,img_path =extra_imitations_path,overwrite=True,save_pre_path = save_pre_path,folders=folders)
    
    
if make_summary:
    rubric.get_summary_df(learner_folder,save = 'default')


if make_viz:
    folder = Path(container,learner_folder)
    plotting.plot_problem_paintings(folder,save=Path(folder,viz),cutoff = cutoff)
    base_path = Path(container,learner_folder,sub_folder)
    for folder in next(os.walk(base_path))[1]:
        plotting.plot_problem_paintings(Path(base_path,folder),save=Path(base_path,folder,viz),cutoff = cutoff)