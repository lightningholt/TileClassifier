from fastai.vision.all import *
from fastai.data.all import *
from utils_fastai import model_run2
import pandas as pd


# arch_name = 'resnet50'
# arch = resnet50
# path = 'Paintings512/Processed/Raw/'
# px_size = 512
# type = 'CNN'

arch_name = 'resnet50'
arch = resnet50
path = 'Paintings/Processed/Raw/'
px_size = 256
type = 'CNN'


# arch_name = 'levit_128'
# arch = arch_name
# path = 'Paintings/Processed/Raw/'
# px_size = 224
# type = 'ViT'

# arch_name = 'densenet121'
# arch = densenet121
# path = 'Paintings/Processed/Raw/'
# px_size = 256
# type = 'CNN'

# arch_name = 'alexnet'
# arch = alexnet
# path = 'Paintings/Processed/Raw/'
# px_size = 256
# type = 'CNN'

# arch_name = 'alexnet'
# arch = alexnet
# path = 'Paintings512/Processed/Raw/'
# px_size = 512
# type = 'CNN'

# arch_name = 'squuezenet1_1'
# arch = squeezenet1_1
# path = 'Paintings/Processed/Raw/'
# px_size = 256
# type = 'CNN'

# arch_name = 'squuezenet1_1'
# arch = squeezenet1_1
# path = 'Paintings512/Processed/Raw/'
# px_size = 512
# type = 'CNN'

# arch_name = arch = 'volo_d5_512'
# arch = arch_name
# path = 'Paintings512/Processed/Raw/'
# px_size = 512
# type = 'ViT'

# arch_name = 'swinv2_cr_tiny_ns_224'
# arch = arch_name
# path = 'Paintings/Processed/Raw/'
# px_size = 224
# type = 'ViT'

# arch_name = 'tiny_vit_21m_512.dist_in22k_ft_in1k'
# arch = arch_name
# path = 'Paintings512/Processed/Raw/'
# px_size = 512
# type = 'ViT'

# arch_name = 'densenet121'
# arch = densenet121
# path = 'Paintings/Processed/Raw/'
# px_size = 256
# type = 'CNN'

# arch_name = 'pvt_v2_b5'
# arch = arch_name
# path = 'Paintings/Processed/Raw/'
# px_size = 224
# type = 'ViT'

bs = 64
epochs = 1

notes = 'blahblah'
computer = 'Desktop'
seed = 72
pretrained= True
add_ho_to_valid=True



folders = ['10','100','105', '110', '115', '120', '125',
 '130', '135', '140', '145', '15', '150', '155', '160',
 '165', '170', '175', '180', '185', '190', '195', '20',
 '200', '205', '210', '215', '220', '225', '230', '235',
 '240', '245', '25', '250', '255', '260', '265', '270',
 '275', '280', '285', '290', '295', '30', '300', '305',
 '310', '315', '320', '325', '330', '335', '340', '345',
 '35', '350', '355', '360', '40', '45', '50',
 '55', '60', '65', '70', '75', '80', '85', '90',
 '95','Max']

# folders = ['Max']

save_str = computer + '-NoAlbumen' +'--' + arch_name + '--'+  'seed-' + str(seed)+ '_bs-' + str(bs) + '_e-' + str(epochs) + '_' +folders[0] + '-' + folders[-1] +'_'

M=pd.read_parquet('master.parquet')

remove_extras = ['P127', 'P128(Left)', 'P128(Right)', 'P129', 'P69(V)']

# valid_list = pd.read_csv('runs/gcp_classifier-3_10-Max_color_10-15-2022/valid_list')
valid_list = pd.read_csv('bidlo_stuff/valid_list')
valid_list = valid_list.valid_list.to_list()
valid_list = [ x for x in valid_list if x not in remove_extras]

print(save_str)
metrics = model_run2(M,
                     save_str,
                     type = type,
                     folders = folders,
                     epochs= epochs,
                     valid_set = valid_list,
                     notes= notes,
                     seed = seed,
                     bs = bs,
                     pretrained= pretrained,
                     arch = arch,
                     path = path,
                     computer = computer,
                     px_size = px_size,
                     add_ho_to_valid=add_ho_to_valid
                     )