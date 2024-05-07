import pandas as pd
import utils_fastai
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL

from fastai.vision.all import *
from fastai.data.all import *

def main():
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
    folders = ['Max']
    metrics = utils_fastai.model_run(M, 'gcp_resnet50', 
                      Descreened = False, 
                      valid_set = 0.2,
                      seed = 42,
                      thread_num = False,
                      computer = 'GCP',
                      px_size = 256,
                      grayscale = False,
                      epochs = 1,
                      model = resnet50,
                      folders = folders,
                      hold_out_catalog= ['93', '165', '179', '207', '231', '801', '820', '367', '371', '380'],
                      hold_out_files = ['A57','A66','A64','A54(R)','C20','C55','C47(R)','D19','D8','E10','F114','F61','F110','F63','F94','F87','F99','G24','G58','G94','G62','G1','G27'],
                      hold_out_duplicates = ['P13(L)','P19(L)','P25(L)','P32(W)','P33(W Lowerqual)',
'P47(F Lowres)','P65(L)','P75(V)','P77(J)','P80(J)','P86(S)','P105(J)','P106(V)','P115(F Lowres)','A14(Left)'],
                      remove_duplicates = True,
                      notes = 'resnet50 run to have on virtual machine'
                       )
    print('finished model run')
    print(metrics)


if __name__ == '__main__':
    main()
    
