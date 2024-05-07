import pandas as pd
import utils_fastai
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL

from fastai.vision.all import *
from fastai.data.all import *

def main():
    learner_folder = 'mac_resnet50_72_40-Max_gray_10-05-2022'
    img_type = 'Descreened'  # or do 'Raw'
    do_valid = True
    do_Bs = True
    do_hold_out = True
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
    img_path = os.path.join('Paintings/Processed/',img_type)
    save_folder_hold_out = learner_folder + '_ho_' + img_type
    save_folder_Bs = learner_folder + '_Bs_' + img_type
    save_folder_valid = learner_folder + '_valid_' + img_type

    M = pd.read_parquet('master.parquet')
    hold_out_catalog= ['93', '165', '179', '207', '231', '801', '820', '367', '371', '380']
    hold_out_files = ['A57','A66','A64','A54(R)','C20','C55','C47(R)','D19','D8','E10','F114','F61','F110','F63','F94','F87','F99','G24','G58','G94','G62','G1','G27']
    hold_out_duplicates = ['P13(L)','P19(L)','P25(L)','P32(W)','P33(W Lowerqual)',
'P47(F Lowres)','P65(L)','P75(V)','P77(J)','P80(J)','P86(S)','P105(J)','P106(V)','P115(F Lowres)','A14(Left)']
    hold_out_files_full = utils_fastai.get_hold_out_files(M,hold_out_catalog,hold_out_files)
    full_save_folder_hold_out = os.path.join('painting_preds',save_folder_hold_out)
    full_save_folder_Bs = os.path.join('painting_preds',save_folder_Bs)
    full_save_folder_valid = os.path.join('painting_preds',save_folder_valid)
    Bs = M[M.file.str.startswith('B')].file.tolist()

    if do_valid:
        if os.path.exists(full_save_folder_valid):
            valid = list(set(pd.read_csv(os.path.join('runs',learner_folder,'valid_list')).valid_list.tolist())-set(os.listdir(full_save_folder_valid)))
        else:
            valid = sorted(list(set(pd.read_csv(os.path.join('runs',learner_folder,'valid_list')).valid_list.tolist())))

        metrics = utils_fastai.get_learn_preds_wrapper(valid, learner_folder, save_folder_valid,img_path = img_path,overwrite=True,folders=folders)

        print('finished valid run')
        print(metrics)
    if do_hold_out:
        if os.path.exists(full_save_folder_hold_out):
            hold_out = sorted(list(set(hold_out_files_full + hold_out_duplicates)-set(os.listdir(full_save_folder_hold_out))))
        else:
            hold_out = sorted(list(set(hold_out_files_full + hold_out_duplicates)))

        metrics = utils_fastai.get_learn_preds_wrapper(hold_out, learner_folder, save_folder_hold_out,img_path = img_path,overwrite=True,folders=folders)
        print('finished hold out run')
        print(metrics)
    if do_Bs:
        if os.path.exists(full_save_folder_Bs):
            hold_out = sorted(list(set(Bs)-set(os.listdir(full_save_folder_Bs))))
        else:
            hold_out = sorted(list(set(Bs)))

        metrics = utils_fastai.get_learn_preds_wrapper(Bs, learner_folder, save_folder_Bs,img_path = img_path,overwrite=True,folders=folders)

        print('finished B run')
        print(metrics)

if __name__ == '__main__':
    main()
