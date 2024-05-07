import make_report_data
import os
import pandas as pd

learner_folder = 'gcp_classifier-3_10-Max_color_10-15-2022'
learner_master = pd.read_parquet('master.parquet')
# file = '3B2A2134_482mp'
# img_path_testing = 'Prometheus/Processed/Raw'
# master_testing = pd.read_parquet('prometheus.parquet')
file = 'A66'
img_path_testing = 'Paintings/Processed/Raw'
master_testing = pd.read_parquet('master.parquet')
# file = 'H2'
# img_path_testing = 'Bristol/Processed/Raw'
# master_testing = pd.read_parquet('master_Bristol.parquet')
make_report_data.make_report_data(master_testing[master_testing.file == file], learner_master,
                                  container = 'runs',
                                  learner_folder = learner_folder,
                                  img_path =  'Paintings/Processed/Raw',
                                  img_path_testing =  img_path_testing,
                                  save_pre_path = 'painting_preds',
                                  save_folder = file+'_' + learner_folder,
                                  do_preds = False,
                                  one_vote = True,
                                  select_painting = 'P90(S)',
                                  round_to = 2)