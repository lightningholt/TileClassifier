import shutil
import hashlib
import os
import argparse
from tqdm import tqdm


def generate_mapping(passphrase, max_int=9):
    # Use a cryptographic hash function to generate a hash from the passphrase
    passphrase_hash = hashlib.sha256(passphrase.encode()).digest()
    # Use the hash to seed the random number generator
    seed = int.from_bytes(passphrase_hash, byteorder='big')
    
    # Generate a deterministic permutation of numbers from 0 to max_int
    mapping = list(range(max_int + 1))
    for i in range(max_int, 0, -1):
        j = seed % (i + 1)
        mapping[i], mapping[j] = mapping[j], mapping[i]
        seed >>= 8  # Shift seed to get the next byte
    
    mapping = dict(zip(range(len(mapping)),mapping))
    return mapping

def map_files(invert_mapping = False,
                passphrase = 'test',
                source_path = 'Paintings/Processed/Raw/',
                target_path = '/mnt/d/adjusted_files/Paintings/Processed/Raw',
                specified_folders = False,
                remove_files_with_prefix = ('B')): #e.g. 
    # if not os.path.exists(target_path):
    #     os.makedirs(target_path)
    if isinstance(specified_folders,list):
        folders = specified_folders
    else:
        folders = [item for item in os.listdir(source_path) if not item in ['Full','Max']]
        folders = sorted(folders, key=lambda x: int(x))
        folders = folders[::-1]

    for folder in folders:
        files = os.listdir(os.path.join(source_path,folder))
        if remove_files_with_prefix:
            files = [item for item in files if not item.startswith(remove_files_with_prefix)]
        file_bases = list(set([file.split('_cropped_')[0] for file in files if file.endswith('tif')]))
        for file_base in tqdm(file_bases, desc=f"Folder {folder}: file_bases completed", unit="files"):
            specified_files = [file for file in files if file.startswith(file_base+'_c')]
            C_list = [int(file.split('.tif')[0].split('_cropped_')[1].split('_')[0][1:]) for file in specified_files]
            R_list = [int(file.split('.tif')[0].split('_cropped_')[1].split('_')[1][1:]) for file in specified_files]
            max_C = max(C_list)
            max_R = max(R_list) 
            C_mapping = generate_mapping(passphrase,max_int = max_C)
            R_mapping = generate_mapping(passphrase,max_int = max_R)
            new_file_strs = []
            if invert_mapping:
                C_mapping = {v: k for k, v in C_mapping.items()}
                R_mapping = {v: k for k, v in R_mapping.items()}
                
            for i, specified_file in enumerate(specified_files):
                new_file_str = file_base + '_cropped_C' + str(C_mapping[C_list[i]]) + '_R' + str(R_mapping[R_list[i]]) + '.tif'
                if not os.path.exists(os.path.join(target_path,folder)):
                    os.makedirs(os.path.join(target_path,folder))
                shutil.copyfile(os.path.join(source_path,folder,specified_file),os.path.join(target_path,folder,new_file_str))

def main():
    parser = argparse.ArgumentParser(description='Process some parameters')
    parser.add_argument('--passphrase', type=str, help = 'passphrase that sets randomization')
    parser.add_argument('--invertmapping', type=bool,help = 'True to bring back to normal')
    parser.add_argument('--source', type=str, help = 'source_path')
    parser.add_argument('--target', type=str, help = 'target_path')
    parser.add_argument('--folders', type=list, help = 'specify specific folders in a list')
    parser.add_argument('--prefix', type=tuple, help = 'string tuple of prefixes to remove')

    args = parser.parse_args()

    if args.passphrase:
        passphrase = args.passphrase
    else:
        passphrase = 'test'

    if args.invertmapping:
        invert_mapping = args.invertmapping
    else:
        invert_mapping = False

    if args.source:
        source_path = args.source
    else:
        source_path = 'Paintings/Processed/Raw/'

    if args.target:
        target_path = args.target
    else:
        target_path = '/mnt/d/adjusted_files/Paintings/Processed/Raw'

    if args.folders:
        specified_folders = args.folders
    else:
        specified_folders = False

    if args.prefix:
        remove_files_with_prefix = args.prefix
    else:
        remove_files_with_prefix = ('B')
    
    map_files(invert_mapping = invert_mapping,
                passphrase = passphrase,
                source_path = source_path,
                target_path = target_path,
                specified_folders = specified_folders,
                remove_files_with_prefix=remove_files_with_prefix)
    
if __name__ == "__main__":
    main()