from fastai.vision.all import *
from fastai.data.all import *
from utils_fastai import model_run2
import pandas as pd
import argparse
# import tempfile
# import os

def str_or_list(s):
    if ',' in s:
        return s.split(',')
    return [s]

def main():
    parser = argparse.ArgumentParser(description='Process some parameters')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--notes', type=str)
    parser.add_argument('--e', type=int, help = 'epochs')
    parser.add_argument('--bs', type=int, help = 'batch size')
    parser.add_argument('--run', type=str, help = 'should be run or max')
    parser.add_argument('--this_arch', type=str_or_list, help = 'overwites the architectures to be executed from run and only runs this one (short name)')
    parser.add_argument('--not_this_arch', type=str_or_list, help = 'overwites the architectures to be executed from run and only runs this one (short name)')
    parser.add_argument('--save_path', type=str, help = 'enter the path where the runs will be saved')
    # parser.add_argument('--skip', type=bool, help = 'skips running the model again')

    args = parser.parse_args()

    if args.run:
        run = args.run
    else:
        run = 'full'
    if args.seed:
        seed = args.seed
    else:
        seed = 72
    if args.bs:
        bs = args.bs
    else:
        bs = 64
    if args.e:
        epochs = args.e
    else:
        epochs = 1
    if args.notes:
        notes = args.notes
        append_notes = True
    else:
        notes = 'blahblah'
    if args.this_arch:
        this_arch = args.this_arch
    else:
        this_arch = None
    if args.not_this_arch:
        not_this_arch = args.not_this_arch
    else:
        not_this_arch = ['']
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = 'runs'
    # if args.skip:
    #     skip = args.skip
    # else:
    #     skip = False
    # run = 'max' #'full'
    # seed = 49 #72
    # bs = 4
    # epochs = 1
    # notes = 'blahblah'

    def set_model(vars_dict):
        for key,value in vars_dict.items():
            globals()[key] = value

    def get_px_size(path,type):
        if path == 'Paintings512/Processed/Raw/':
            return 512
        elif path == 'Paintings/Processed/Raw/':
            if type == 'ViT':
                return 224
            else:
                return 256
        else:
            print('unsupported path!')

    runs = {
        'max': {
            'models' : ['resnet50','resnet34','resnet101','xresnet50_d','xresnet50','xresnext50','alexnet','squeezenet','efficientnet','tiny_vit','maxvit','volo'],
            'path' : 'Paintings512/Processed/Raw/',
            'folders' : ['Max']
        },
        'full': {
            'models' : ['resnet50','resnet34','resnet101','alexnet','squeezenet','densenet','swin','pvt','levit'],
            'path' : 'Paintings/Processed/Raw/',
            'folders' : ['10','100','105', '110', '115', '120', '125',
                '130', '135', '140', '145', '15', '150', '155', '160',
                '165', '170', '175', '180', '185', '190', '195', '20',
                '200', '205', '210', '215', '220', '225', '230', '235',
                '240', '245', '25', '250', '255', '260', '265', '270',
                '275', '280', '285', '290', '295', '30', '300', '305',
                '310', '315', '320', '325', '330', '335', '340', '345',
                '35', '350', '355', '360', '40', '45', '50',
                '55', '60', '65', '70', '75', '80', '85', '90',
                '95','Max']
        }
    }

    models = { 
        'resnet50': {
                'arch_name' : 'resnet50',
                'arch' : resnet50,
                'type' : 'CNN'
        },
        'alexnet':{
            'arch_name' : 'alexnet',
            'arch' : alexnet,
            'type' : 'CNN'
        },
        'squeezenet':{
            'arch_name' : 'squuezenet1_1',
            'arch' :  squeezenet1_1,
            'type' : 'CNN'
        },
        'tiny_vit':{
            'arch_name' : 'tiny_vit_21m_512.dist_in22k_ft_in1k',
            'arch' :  'tiny_vit_21m_512.dist_in22k_ft_in1k',
            'type' : 'ViT'
        },
        'maxvit':{
            'arch_name' : 'maxvit_base_tf_512.in1k',
            'arch' :  'maxvit_base_tf_512.in1k',
            'type' : 'ViT'
        },
        'xresnet50_d':{
            'arch_name' : 'xresnet50_deep',
            'arch' :  xresnet50_deep,
            'type' : 'CNN'
        },
        'xresnext50':{
            'arch_name' : 'xresnext50',
            'arch' :  xresnext50,
            'type' : 'CNN'
        },
        'xresnet50':{
            'arch_name' : 'xresnet50',
            'arch' :  xresnet50,
            'type' : 'CNN'
        },
        'resnet34':{
            'arch_name' : 'resnet34',
            'arch' :  resnet34,
            'type' : 'CNN'
        },
        'resnet101':{
            'arch_name' : 'resnet101',
            'arch' :  resnet101,
            'type' : 'CNN'
        },
        'efficientnet':{
            'arch_name' : 'efficientnet_b5.sw_in12k',
            'arch' :  'efficientnet_b5.sw_in12k',
            'type' : 'CNN'
        },
        'volo':{
            'arch_name' : 'volo_d5_512',
            'arch' :  'volo_d5_512',
            'type' : 'ViT'
        },
        'levit':{
            'arch_name' : 'levit_128',
            'arch' :  'levit_128',
            'type' : 'ViT'
        },
        'swin':{
            'arch_name' : 'swinv2_cr_tiny_ns_224',
            'arch' :  'swinv2_cr_tiny_ns_224',
            'type' : 'ViT'
        },
        'pvt':{
            'arch_name' : 'pvt_v2_b5',
            'arch' :  'pvt_v2_b5',
            'type' : 'ViT'
        },
        'densenet':{
            'arch_name' : 'densenet121',
            'arch' :  densenet121,
            'type' : 'CNN'
        },
    }


    computer = 'Desktop'

    pretrained= True
    add_ho_to_valid=True



        # folders = ['10','100','105', '110', '115', '120', '125',
        #  '130', '135', '140', '145', '15', '150', '155', '160',
        #  '165', '170', '175', '180', '185', '190', '195', '20',
        #  '200', '205', '210', '215', '220', '225', '230', '235',
        #  '240', '245', '25', '250', '255', '260', '265', '270',
        #  '275', '280', '285', '290', '295', '30', '300', '305',
        #  '310', '315', '320', '325', '330', '335', '340', '345',
        #  '35', '350', '355', '360', '40', '45', '50',
        #  '55', '60', '65', '70', '75', '80', '85', '90',
        #  '95','Max']

    # folders = ['Max']

    #choose the models based on the type of run 'run'
    selected_models = {}
    if this_arch:
        for model_name in this_arch:
            selected_models[model_name] = models[model_name]
    else:
        for model_name in runs[run]['models']:
            if model_name not in not_this_arch:
                selected_models[model_name] = models[model_name]



    


    folders = runs[run]['folders']
    path = runs[run]['path']

    M=pd.read_parquet('master.parquet')
    valid_list = pd.read_csv('runs/gcp_classifier-3_10-Max_color_10-15-2022/valid_list')
    valid_list = valid_list.valid_list.to_list()

    for model_name,model_vars in selected_models.items():
        px_size = get_px_size(path,model_vars['type'])
        # if model_name != 'volo':
        set_model(model_vars)
        if append_notes:
            save_str = computer +'--' + arch_name + '--'+  'seed-' + str(seed)+ '_bs-' + str(bs) + '_e-' + str(epochs) + '_' +folders[0] + '-' + folders[-1] +'_' + notes + '_'
        else:
            save_str = computer +'--' + arch_name + '--'+  'seed-' + str(seed)+ '_bs-' + str(bs) + '_e-' + str(epochs) + '_' +folders[0] + '-' + folders[-1] +'_'
        # print(skip)
        # stuff_in_path = [item.split('__')[0] for item in os.listdir(save_path)]
        # if os.path.exists(save_str) and skip:
        #     print(save_str + 'already exists, skipping')
        # else:
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
                            add_ho_to_valid=add_ho_to_valid,
                            pre_path = save_path)
        

if __name__ == "__main__":
    main()
            
