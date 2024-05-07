import pandas as pd
import shutil
import os
from pathlib import Path
import numpy as np

def test_folders(master,base_folder = 'Paintings/Processed/Raw/',full_paint_folder = 'Full',verbose = False,files_from_dir = False,center_tile = True):
    inconsistent = []
    file_names = master.file.tolist()
    output_sizes_cm = sorted([int(item) for item in os.listdir(base_folder) if item.isnumeric()])
    for file_name in file_names:
        if verbose:
            print('checking consistency for ', file_name)
        if master['file'].isin([file_name]).any():
            resolution = float(master.loc[master['file'] == file_name]['px_per_cm_height'])
            for output_size_cm in output_sizes_cm:
                output_size_px = round(output_size_cm * resolution)
                expect_num_slices = count_tile_from_master(master, file_name,output_size_px,center_tile=center_tile)
                slices = os.listdir(os.path.join(base_folder,str(output_size_cm)))
                folder_num_slices = len([item for item in slices if item.startswith(file_name + '_c')])
                if expect_num_slices != folder_num_slices:
                    if output_size_cm < min(master[master.file == file_name].height_cm.iloc[0],master[master.file == file_name].width_cm.iloc[0]):
                        inconsistent.append({'file':file_name,'expected':expect_num_slices,'in_folder':folder_num_slices,'folder_size':output_size_cm})
                        if verbose:
                            print('expected',expect_num_slices,'folder',folder_num_slices,' for', file_name, 'in',output_size_cm)
        else:
            print('...master does not contain file_name ', file_name)

    return inconsistent

def count_tile_from_master(master, file_name,output_size,center_tile = True):
    output_size = [output_size,output_size] #throw in to SliceCrop class to get output_size formatted

    h = master[master.file == file_name].height_px.iloc[0]
    w = master[master.file == file_name].width_px.iloc[0]
    px_dims = [h,w]
    if center_tile:
        num_slices =[int(px_dims[i]/output_size[i]) for i in (0,1)]   
    else:
        num_slices = [int(np.ceil(px_dims[i]/output_size[i]))+0 if output_size[i] <= px_dims[i] else 0 for i in (0,1) ]
    
    return num_slices[0]*num_slices[1]

def remove_files_with_startstring(start_string, root):
    for path, subdirs, files in os.walk(root):
        # print(path)
        for name in files:
            # get file path 
            file_path = os.path.join(path, name)
            if name.split('_cropped')[0] == start_string:
                if Path(file_path).parent.stem != 'Full':
                    os.remove(file_path)
                    # print(file_path)
def move_files_with_startstring(start_string, root,dst_folder,test = False):
    for path, subdirs, files in os.walk(root):
        # print(path)
        for name in files:
            # get file path 
            file_path = os.path.join(path, name)
            if name.split('_cropped')[0] == start_string:
                if Path(file_path).parent.stem != 'Full':
                    # print(path)
                    # print(file_path.split(root)[1])
                    dst_path = os.path.join(dst_folder,file_path.split(root)[1])
                    if not os.path.exists(Path(dst_path).parent):
                        # print(Path(dst_path).parent)
                        os.mkdir(Path(dst_path).parent)
                    if test:
                        print(file_path,dst_path)
                    else:
                        shutil.move(file_path, dst_path)
def remove_files_with_string(string, 
                             root,
                             omit_parent = False #'Full'
                             ):
    # omit_parent = 'Full'
    for path, subdirs, files in os.walk(root):
        print(path)
        for name in files:
            # get file path 
            file_path = os.path.join(path, name)
            if string in name:
                if omit_parent:
                    if Path(file_path).parent.stem != omit_parent:
                        os.remove(file_path)
                        # print(file_path)
                else:
                    os.remove(file_path)