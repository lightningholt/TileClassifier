from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
gauth = GoogleAuth()           
drive = GoogleDrive(gauth)  

def get_file_list_dict(folder_id):
    file_list = drive.ListFile({'q':"'{}' in parents and trashed=false".format(folder_id)}).GetList()
    dir_dict = {file1['title']:file1['id'] for file1 in file_list}
    return dir_dict

def rename_file_gdrive(new_name, file_id):
    # CreateFile() can be called with an existing id.
    file1 = drive.CreateFile({'id': file_id})
    file1.FetchMetadata()
    file1['title'] = new_name # Change title.
    file1.Upload() 
    
def rename_slice_folder_gdrive(folder_id):
    dir_dict = get_file_list_dict(folder_id)
    folder = drive.CreateFile({'id': folder_id})
    # folder.FetchMetadata()
    slice_size = folder['title']
    for key in dir_dict:
        file_name = key
        file_id = dir_dict[key]
        insertion = 'S' + slice_size
        new_file_name = get_new_slice_name(file_name,insertion)
        if file_name != new_file_name:
            rename_file_gdrive(new_file_name,file_id)
            
def rename_slice_folder_gdrive_delete(folder_id):
    dir_dict = get_file_list_dict(folder_id)
    folder = drive.CreateFile({'id': folder_id})
    # folder.FetchMetadata()
    slice_size = folder['title']
    for key in dir_dict:
        file_name = key
        file_id = dir_dict[key]
        deletion = '_S' + slice_size
        new_file_name = get_new_slice_name_delete(file_name,deletion)
        if file_name != new_file_name:
            rename_file_gdrive(new_file_name,file_id)            

def get_new_slice_name(slice_name, insertion, split = 'd_C',insert_on = '_'):
    if len(slice_name.split(split))==2:
        new_slice_name = slice_name.split(split)[0] + split.split(insert_on)[0] + insert_on + insertion + insert_on + split.split(insert_on)[1]+slice_name.split(split)[1]
        return new_slice_name
    else:
        return slice_name
    
def get_new_slice_name_delete(slice_name, delete_string):
    if len(slice_name.split(delete_string))==2:
        new_slice_name = slice_name.split(delete_string)[0] + slice_name.split(delete_string)[1]
        return new_slice_name   
    else:
        return slice_name
    
def rename_slices_local(read_dir, slice_size):
    slice_size = str(slice_size)
    files = os.listdir(os.path.join(read_dir,slice_size))
    paths = [os.path.join(read_dir,slice_size,file) for file in files]
    for path in paths:
        if len(path.split('d_C'))==2:
            print(len(path.split('d_C')))
            os.rename(path, os.path.join(path.split('d_C')[0] + 'd_S'+ slice_size + '_C' + path.split('d_C')[1]))
            
def upload_file_to_gdrive(file_dir, file_name, folder_id):
    gfile = drive.CreateFile({'parents': [{'id': folder_id}]})
    gfile.SetContentFile(os.path.join(file_dir,file_name))
    gfile['title'] = file_name
    gfile.Upload() # Upload the file.        
    
def upload_files_to_gdrive(read_dir,folder_id,rewrite = False):
    print('Fetching local files')
    file_names = os.listdir(read_dir)
    print('{} local files'.format(len(file_names)))
    if not rewrite:
        print('Fetching gdrive files')
        gdrive_files = get_file_list_dict(folder_id)
        print('{} gdrive files'.format(len(gdrive_files)))
        print('Checking difference in folders')
        file_names = list(set(file_names).difference(set(gdrive_files)))
        if len(file_names) == 0:
            print('Folder is up to date')
    print('uploading {} files'.format(len(file_names)))
    pbar = tqdm(total = len(file_names))
    for file_name in file_names:
        pbar.update()
        upload_file_to_gdrive(read_dir,file_name,folder_id)
    pbar.close()
        
def create_folder(folder_name,parent_id):
    drive = GoogleDrive(gauth)

    folder_metadata = {
        'title' : folder_name, 
        'parents': [{'id':parent_id}],
        'mimeType' : 'application/vnd.google-apps.folder'
    }
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    
def download_files_from_gdrive_folder(folder_id,save_location):
    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(folder_id)}).GetList()
    pbar = tqdm(total = len(file_list))
    for i, file1 in enumerate(sorted(file_list, key = lambda x: x['title']), start=1):
        pbar.update()
        # print('Downloading {} from GDrive ({}/{})'.format(file1['title'], i, len(file_list)))
        full_path = os.path.join(save_location,file1['title'])
        file1.GetContentFile(full_path)
    pbar.close()   