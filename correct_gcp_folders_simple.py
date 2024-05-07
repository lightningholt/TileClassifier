import pandas as pd
import shutil
import os
from pathlib import Path

def test_folders(master,base_folder = 'Paintings/Processed/Raw/',full_paint_folder = 'Full',verbose = False,files_from_dir = False):
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
                expect_num_slices = count_tile_from_master(master, file_name,output_size_px)
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

def count_tile_from_master(master, file_name,output_size):
    output_size = [output_size,output_size] #throw in to SliceCrop class to get output_size formatted

    h = master[master.file == file_name].height_px.iloc[0]
    w = master[master.file == file_name].width_px.iloc[0]
    px_dims = [h,w]
    num_slices =[int(px_dims[i]/output_size[i]) for i in (0,1)]   
    
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

#define the things that might need changing
main_raw_path = 'Paintings/Processed/Raw/'
main_descreened_path = 'Paintings/Processed/Descreened/'

files_to_redo_2_R_path = 'Paintings/Processed/Raw/Raw2/'
files_to_redo_2_D_path = 'Paintings/Processed/Raw/Descreened2/'

files_to_redo_3_R_path = 'Paintings/Processed/Raw/Raw3/'
files_to_redo_3_D_path = 'Paintings/Processed/Raw/Descreened3/'

files_to_redo_4_R_path = 'Paintings/Processed/Raw/A8_R/'
files_to_redo_4_D_path = 'Paintings/Processed/Raw/A8_D/'


master = pd.read_parquet('master.parquet')
M = master[master['px_per_cm_height'].notnull()]

#define all the problem files

files_to_redo_2 = ['A9(right)',
 'B16',
 'B24',
 'B25',
 'B40',
 'C57',
 'C61',
 'F111',
 'F18',
 'F20',
 'F31',
 'F38',
 'F60',
 'F80',
 'G1',
 'G16',
 'G3(R)',
 'G34',
 'G6',
 'JPCR_00185',
 'JPCR_00186',
 'JPCR_00194',
 'JPCR_00204',
 'JPCR_00205',
 'JPCR_00208',
 'JPCR_00217',
 'JPCR_00220(Middle)',
 'JPCR_00221(Right)',
 'JPCR_00222',
 'JPCR_00223',
 'JPCR_00240',
 'JPCR_00245',
 'JPCR_00249',
 'JPCR_00251',
 'JPCR_00262',
 'JPCR_00269',
 'JPCR_00271',
 'JPCR_00272',
 'JPCR_00274',
 'JPCR_00283',
 'JPCR_00297',
 'JPCR_00312',
 'JPCR_00358',
 'JPCR_00363',
 'JPCR_00367',
 'JPCR_00792',
 'JPCR_01002',
 'JPCR_01019',
 'JPCR_01030',
 'JPCR_01088',
 'P106(L Left)',
 'P106(L Right)',
 'P106(V)',
 'P117(F)',
 'P119(B)',
 'P42(W)',
 'P62(F)',
 'P63(S)',
 'P67(F)']

files_to_redo_3 = ['JPCR_00263','JPCR_00301','G52']

files_to_redo_4 = ['A8']
all_files = files_to_redo_2 + files_to_redo_3 + files_to_redo_4

#first we remove the problem files from the main painting folders
print('removing problem slices')
for file in all_files:
    remove_files_with_startstring(file,main_raw_path)
    remove_files_with_startstring(file,main_descreened_path)

print('moving in problem slices (this might take a while)')
#then we move new nice folders from corrected path into the main painting folders
for file in files_to_redo_2:
    move_files_with_startstring(file,files_to_redo_2_R_path,main_raw_path)
    move_files_with_startstring(file,files_to_redo_2_D_path,main_descreened_path)
    
print('moving in problem slices (much shorter now)')   
for file in files_to_redo_3:
    move_files_with_startstring(file,files_to_redo_3_R_path,main_raw_path)
    move_files_with_startstring(file,files_to_redo_3_D_path,main_descreened_path)
    
print('moving in problem slices (much shorter now)')   
for file in files_to_redo_4:
    move_files_with_startstring(file,files_to_redo_4_R_path,main_raw_path)
    move_files_with_startstring(file,files_to_redo_4_D_path,main_descreened_path)
    
print('checking to make sure everything is svelt...')
raw_inconsistent = test_folders(M,base_folder = main_raw_path)
print('inconsistent raw files ',raw_inconsistent)
descreened_inconsistent = test_folders(M,base_folder = main_raw_path)
print('inconsistent descreened files ',descreened_inconsistent)