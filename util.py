import numpy as np
from PIL import Image
import os
import glob
import cv2
from fastai.vision.all import *
from fastai.data.all import *
import fastparquet
from slice_classes import AbstractArtData, SliceCrop, Rescale
from torchvision import transforms, utils
import imageio
import pandas as pd
from ast import literal_eval
import math
from PIL import ImageOps, ImageDraw



def scale_images(pix_size, dir = 'all_images/Pollock'):
    basewidth = int(np.sqrt(pix_size))
    type = dir.split('/')[-1]

    for infile in glob.glob(dir + '/*.tif'):
        im = Image.open(infile)
        fname = infile.split('/')[-1]
        wpercent = (basewidth/float(im.size[0]))
        hsize = int((float(im.size[1])*float(wpercent)))
        img = im.resize((basewidth,hsize), Image.ANTIALIAS)
        img.save('Rescaled_images/'+type+'/'+fname)

def crop_images(crop_size, fname):
    '''
    function to just-in-time crop images for input to NN

    Inputs:
    crop_size = side length of desired new image
    fname = file name of image to be cropped (include path)
    '''

    im = Image.open(fname)
    width, height = im.size   # Get dimensions

    left = (width - crop_size)/2
    top = (height - crop_size)/2
    right = (width + crop_size)/2
    bottom = (height + crop_size)/2

    #crop center of image
    im = im.crop((left, top, right, bottom))

    return im

def waitDestroyAllWindows():
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
def getBoundingCoords(dir_name,file):  
    path = os.path.join(dir_name,file)
    img = cv2.imread(path)
    coordinateStore1 = CoordinateStore(img)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', coordinateStore1.select_point)
    cv2.waitKey(0)
    waitDestroyAllWindows()
    return coordinateStore1.points    

def getAllBoundingCoords(dir_name,files,save=True,save_name = 'cnts',save_type = 'parquet',append_file = True):
    cnts = []
    # cont = True
    cnt_dict = {}
    files_done = []
    save_ext = '.' + save_type
    save_path = save_name + save_ext #os.path.join(dir_name,save_name + save_ext)
    for file in files:
        coords = getBoundingCoords(dir_name,file)
        if len(coords) == 4:
            cnts.append(coords)
            # print(type(cnts),cnts)
            files_done.append(file)
        elif len(coords) == 1: #returns original coords
            cnts.append(getOriginalCoords(dir_name,file))
            files_done.append(file)
        else:
            print('Exiting because number of coordinates was equal to ', len(coords), ' (not 4 or 1)')      
            break     
    
    # cnt_dict = {'file':files_done,'savePath':dir_name,'refPts':cnts}
    zipped_list = list(zip(files_done,[str(dir_name)]*len(cnts),cnts))
    # print(save_path,type(save_path),dir_name,type(str(dir_name)))
    # print(type(cnt_dict['refPts']),cnt_dict['refPts'])
    # print(type(cnts[0]))
    columns = ['file','savePath','refPts']
    df = pd.DataFrame(zipped_list,columns = columns)
    # print(cnt_dict)
    # df = pd.DataFrame(cnt_dict)
    if save:
        if append_file:
            if os.path.exists(save_path):
                df2 = pd.read_parquet(save_path)
                df = pd.concat([df2,df])
                df = df.drop_duplicates(subset=['file'], keep='last')
            else:
                df = df
        else:
            df = df
        # print(result)
        df = df[columns]
        
        df.to_parquet(save_path)
        print('Saved to :',save_path)
    df.reset_index(inplace=True)
    df = df[['file','savePath','refPts']]
    return df

class CoordinateStore:
    def __init__(self,img):
        self.points = []
        self.img = img

    def select_point(self,event,x,y,flags,param):
            # if event == cv2.EVENT_LBUTTONDBLCLK:
            #     cv2.circle(img,(x,y),3,(255,0,0),-1)
            #     self.points.append((x,y))
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append([[x,y]])
                # displaying the coordinates
                # on the image window
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(self.img, str(x) + ',' +
                            str(y), (x,y), font,
                            5, (255, 0, 0), 5)
                cv2.imshow('image', self.img)

def crop_from_corners(img,coords):
    cnt = np.array([
        coords[0],
        coords[1],
        coords[2],
        coords[3]
    ])
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # get width and height of the detected rectangle
    # print(rect)
    width = int(rect[1][0])
    height = int(rect[1][1]) 
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # print(M)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    
    # print(rect[2])
    if 45-rect[2] < 0: #corrects for when the painting is tilted counter clockwise.
        warped = np.rot90(warped,k=3,axes=(0,1))
        (width,height) = (height,width)

    return warped,[width,height]

def master_crops(dfl,write_dir,save=True,save_name = 'dims',save_type = 'parquet',append_file = True):
    dims = []
    files = []
    for i, row in dfl.iterrows():
        path = os.path.join(row.savePath,row.file)
        coords = row.refPts
        print(row.file)
        # print(os.getcwd(),path)
        img = cv2.imread(path)
        # print(img)
        image,dim =crop_from_corners(img,coords)
        dims.append(dim)
        write_file = os.path.splitext(row.file)[0]+"_cropped"+".tif"
        files.append(write_file)
        write_path = os.path.join(write_dir,write_file)
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        cv2.imwrite(write_path, image)
        # print("saved to ",write_path)
    zipped_list = list(zip(files,[str(write_dir)]*len(files),dims))
    columns = ['file','savePath','dims(w,h)']
    df = pd.DataFrame(zipped_list,columns = columns)
    
    save_ext = '.' + save_type
    save_path = save_name + save_ext    
    if save:
        if append_file:
            if os.path.exists(save_path):
                df2 = pd.read_parquet(save_path)  
                df = pd.concat([df2,df])
                df = df.drop_duplicates(subset=['file'], keep='last')
            else:
                df = df
        else:
            df = df
        # print(result)
        df = df[columns]        
        df.to_parquet(save_path)
        
    df.reset_index(inplace=True)
    return df 

def getOriginalCoords(dir_name,file):  
    path = os.path.join(dir_name,file)
    img = cv2.imread(path)
    shape = img.shape
    height = shape[0]
    width = shape[1]
    coords = [[[0, 0]],[[width-1, 0]],[[width-1, height-1]],[[0, height-1]]]
    # coordinateStore1 = CoordinateStore(img)
    # cv2.imshow('image', img)
    # cv2.setMouseCallback('image', coordinateStore1.select_point)
    # cv2.waitKey(0)
    # waitDestroyAllWindows()
    return coords


# test_dataset = AbstractArtData(root_dir='Paintings/Processed/Descreened/Full/')
# slices = tile_an_image(test_dataset[0],1000)
def tile_an_image(sample, output_size, center_tile = True ,rescale = True,write_dir = None,image_size = 256,add = 0,verbose = True):
    stem = Path(sample['name']).stem
    # print(write_dir)
    # print(output_size)
    output_size = SliceCrop(int(output_size),0).output_size #throw in to SliceCrop class to get output_size formatted
    
    if center_tile:
        start_loc = get_tile_start(sample,output_size)
        num_slices = [int(sample['image'].shape[i]/output_size[i]) for i in (0,1)]
    else:
        start_loc = (0,0)
        num_slices, slice_locs_0, slice_locs_1 = get_num_tiles(sample['image'].shape,output_size,add=add)

        
    sc = SliceCrop(output_size,start_loc)

    print_if(verbose,num_slices)
    # slice_locs = []
    # slices = []
    scaled_slices = []
    
    if rescale:
        composed = transforms.Compose([sc,Rescale(image_size)])
    else:
        composed = transforms.Compose([sc])    
        
    if write_dir and (np.min(num_slices) > 0):
        if not center_tile:
            if add > 0:
                write_dir = str(Path(str(Path(write_dir).parent)+'-overlap'+'-'+str(add),Path(write_dir).stem))
            else:
                write_dir = str(Path(str(Path(write_dir).parent)+'-overlap',Path(write_dir).stem))
            # print(write_dir)
        if not os.path.exists(Path(write_dir).parent):
            os.mkdir(Path(write_dir).parent)           
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
            
    
    # slice_loc = start_loc
    for i in range(num_slices[0]):
        for j in range(num_slices[1]):
            if center_tile:
                slice_loc = (start_loc[0] + output_size[0]*i,start_loc[1] + output_size[1]*j)
            else:
                slice_loc = (slice_locs_0[i],slice_locs_1[j])
                # print(slice_loc)
            # slice_locs.append(slice_loc)
            sc.borders = slice_loc
            slc = composed(sample)
            scaled_slices.append(slc)
            if write_dir:
                location_tag ='_C'+str(i).zfill(len(str(num_slices[0])))+'_R'+str(j).zfill(len(str(num_slices[1])))
                full_path = os.path.join(write_dir,stem+location_tag+'.tif')
                print_if(verbose,full_path)
                imageio.imwrite(full_path,slc['image'])
                
            
            
    # if rescale:
    #     scale = Rescale(256)
    #     scaled_slices = []
    #     for slice in slices:
    #         scaled_slices.append(scale(slice))
            
    # if write_dir:  # How do we want to write these? Particularily the naming convention.
    
    
        
        
    
    return scaled_slices

def get_num_tiles(im_shape,output_size,num_slices = False,add = 0):
    assert isinstance(output_size,tuple), 'output_size must be a tuple'
    if not num_slices:
        num_slices = [int(np.ceil(im_shape[i]/output_size[i]))+add if output_size[i] <= im_shape[i] else 0 for i in (0,1) ]
    slice_locs_0 = np.linspace(0, im_shape[0] - output_size[0],num_slices[0],dtype = int) 
    slice_locs_1 = np.linspace(0, im_shape[1] - output_size[1],num_slices[1],dtype = int)
    return num_slices, slice_locs_0,slice_locs_1

def set_tile_on_nan_matrix(im_size,tile_matrix,start_loc):

    # Define the larger matrix
    im_matrix = np.full(im_size,np.nan)  # Example 5x5 matrix of zeros

    # Calculate the end location based on the size of the smaller matrix
    end_loc = [start_loc[i] + tile_matrix.shape[i] for i in (0,1)]

    # Perform element-wise multiplication
    im_matrix[start_loc[0]:end_loc[0], start_loc[1]:end_loc[1]] = tile_matrix

    # Print the updated larger matrix
    return im_matrix

def get_tile_start(sample,output_size):
    if isinstance(sample,tuple):
        h, w = sample
    else:
        if isinstance(sample,np.ndarray):
            image = sample
        else:
            image = sample['image']

        h, w = image.shape[:2]
    top_border = round((h % output_size[0])/2)
    left_border = round((w % output_size[1])/2)
    return (top_border,left_border) 


def filter_to_big_enough_images(df, output_size, end_str='_cm'):
    tmp = df.copy()
    tmp.loc[:, 'big_enough_height'] = tmp['height'+end_str] > output_size
    tmp['big_enough_width'] = tmp['width'+end_str] > output_size
    tmp['fits_box'] = (tmp['big_enough_height']) & (tmp['big_enough_width'])
    df = df[tmp['fits_box']]
    # df.drop(['big_enough_height', 'big_enough_width', 'fits_box'], axis=1, inplace=True)
    return df

def get_fitting_samples(master, samples, output_size_cm,verbose = True):
    Mb = filter_to_big_enough_images(master, output_size_cm)
    # print(Mb)
    fitting_items = tuple([item +'_cropped' for item in Mb.file.tolist()])
    # print('fitting_items=',fitting_items)
    fitting_files = [sample for sample in samples.image_names if sample.startswith(fitting_items)]
    # print('image_names=',samples.image_names)
    print_if(verbose,fitting_files)
    return fitting_files

def tile_images_from_dir(read_dir, master, output_size_cm, redo = False, selected = None,update = True, image_size = 256,center_tile = True,add = 0,verbose = True):
    #General use: Does not overwrite files in directories
    #Ex. tile_images_from_dir('Paintings/Processed/Raw/Full/',M, output_size_cm)
    
    #Overwrite files:
    #Ex. tile_images_from_dir('Paintings/Processed/Raw/Full/',M, output_size_cm, update = False)
    
    #Run for selected files: Does overwrite files
    #Ex. tile_images_from_dir('Paintings/Processed/Raw/Full/',M, output_size_cm,selected =selected_tifs)
    failed_to_tile = []
    # print(read_dir)
    samples = AbstractArtData(root_dir=read_dir)
    write_dir = Path(Path(read_dir).parents[0],str(output_size_cm))
    if update and os.path.isdir(write_dir):
        # print(read_dir,write_dir)
        samples.image_names = get_dir_core_differences(read_dir,write_dir)
    if selected: #just grab a few selected files to tile
        if isinstance(selected,str):
            samples.image_names = [selected]
        else:
            samples.image_names = selected
    
    master = filter_to_big_enough_images(master, output_size_cm) # should only have big enough images now
    # samples.image_names = master.file.tolist()
    # print(master.file)
    # samples = samples.image_names
    # samples = [sample.split('_cropped')[0] for sample in samples.image_names]
    # print(samples.image_names)
    samples.image_names = get_fitting_samples(master, samples, output_size_cm,verbose = verbose)
    #print('image_names',samples.image_names)
    
    for sample in samples:     
        # file_name = sample
        file_name = sample['name'].split('_cropped')[0]
        #print(sample['name'])
        #print('checking', file_name)
        if file_name in master['file'].unique():
        # print(master.loc[master['file'] == file_name]['px_per_cm_height'])
            # print(file_name)
            resolution = float(master.loc[master['file'] == file_name]['px_per_cm_height'])
            # print(file_name, ' fits at least one box inside it')
            output_size_px = round(output_size_cm * resolution)
            # print(sample)
            tile_an_image(sample,output_size_px,write_dir=write_dir, image_size=image_size,center_tile = center_tile,add = add,verbose = verbose)
            # else:
            #     print(file_name, 'was skipped because box bigger than a dimension')
        else:
            failed_to_tile.append(file_name)
            # print('Failed to tile: ',file_name, ' at crop size: ', output_size_cm)
    if len(failed_to_tile)>0:
        print('Failed to tile: ',failed_to_tile, ' at crop size: ', output_size_cm)
            
def get_dir_core_differences(read_dir,write_dir,split_str='_cropped'):
    #get list of files in read dir
    read_files = [file for file in os.listdir(read_dir) if not file.startswith('.')]
    read_cores = set([file.split(split_str)[0] for file in read_files])
    # get list of file stems in write dir
    write_files = [file for file in os.listdir(write_dir) if not file.startswith('.')]
    write_cores = set([file.split(split_str)[0] for file in write_files])
    #find difference (only non hidden files)
    file_diff = [file for file in list(read_cores-write_cores)]
    file_diff_appended = [file + split_str for file in file_diff]

    new_read_files = [i for i in read_files if any(b in i for b in file_diff_appended)]
    new_data = [list(b) for a, b in itertools.groupby(new_read_files, key=lambda x: x.split("-")[0])]
    final_data = [random.choice(i) for i in new_data]
    
    return final_data


def save_master_files(master_df, file_name=None):
    if file_name is None:
        file_name = 'master'
    
    master_df.to_csv(file_name + '.csv')
    master_df.to_parquet(file_name + '.parquet')
    
    
def tile_images_from_dir_max_crop(read_dir, master, redo = False, selected = None,update = True,image_size=256,write_folder = 'Max',center_tile = True,add = 0,verbose = True):
    failed_to_tile = []
    
    samples = AbstractArtData(root_dir=read_dir)
    write_dir = Path(Path(read_dir).parents[0],write_folder)
    # if update and os.path.isdir(write_dir):
    #     # print(read_dir,write_dir)
    #     samples.image_names = get_dir_core_differences(read_dir,write_dir)
    #     core_diffs = get_dir_core_differences(read_dir,write_dir)
    
    if selected: #just grab a few selected files to tile
        if isinstance(selected,str):
            samples.image_names = [selected]
        else:
            samples.image_names = selected
    else:
        if update and os.path.isdir(write_dir):
            core_diffs = get_dir_core_differences(read_dir,write_dir)
            samples.image_names = [item + '_cropped.tif' for item in list(master.file) if item + '_cropped.tif' in core_diffs]
        else:
            samples.image_names = [item + '_cropped.tif' for item in list(master.file)]

    for sample in samples:     
        file_name = sample['name'].split('_cropped')[0]
        if file_name in master['file'].unique():
            # print(file_name)
            print_if(verbose,file_name, ' fits at least one box inside it')
            output_size_px = np.min((master.loc[master['file'] == file_name]['height_px'],master.loc[master['file'] == file_name]['width_px']))
            # print(sample)
            tile_an_image(sample,output_size_px,write_dir=write_dir,image_size=image_size,center_tile= center_tile,add = add,verbose = verbose)
        else:
            failed_to_tile.append(file_name)
    if len(failed_to_tile)>0:
        print('Failed to tile: ',failed_to_tile)
        
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
                    
def index_containing_substring(the_list, substring):
    index = []
    for i, s in enumerate(the_list):
        if isinstance(s,str):
            if substring in s:
                index.append(i)
    return index

def df_field_contains_substring(df,field,substring):
    return df.iloc[index_containing_substring(df[field], substring)]

def false_visulizer(compiled_results_false,folders,savefolder,image_base_path = 'Paintings/Processed/Descreened'):
    false_list=compiled_results_false['painting'].tolist()
    for i in false_list:
        false_painting=i
        print(i)
        for i in folders:
            print (i) 
            final_image=image_visualizer(results,i,false_painting,false_painting+'_'+i+'.jpg',image_base_path
                                         ,savepath = savefolder,transparency = False)
def vote_system(results, one_vote=True, binarize=False, binarize_prob=0.5, decision_prob=0.56):  
    if 'set' in results.columns.tolist():
        one_vote_columns = ["painting",'slice_size',"set"]
        gen_columns = ["painting","set"]
    else:
        one_vote_columns = ["painting",'slice_size']
        gen_columns = ["painting"]
    paintingname=results.apply(lambda row: (row.file.split('_c')[0]),axis=1)
    results['painting']=paintingname
    results['prediction'] = np.where(results.pollock_prob>=binarize_prob, 1, 0)
    if one_vote==True:        
        results=results.groupby(one_vote_columns).agg({
                             "prediction":np.mean,"pollock_prob":np.mean,"actual":np.mean}).reset_index()
        results.rename(columns = {'prediction':'prediction_percentage'}, inplace = True)
        results=results.groupby(gen_columns).agg({
                                         "prediction_percentage":np.mean,"pollock_prob":np.mean,"actual":np.mean}).reset_index()
    elif one_vote==False:
        results=results.groupby(gen_columns).agg({
                                         "prediction":np.mean,"pollock_prob":np.mean,"actual":np.mean}).reset_index()
        results.rename(columns = {'prediction':'prediction_percentage'}, inplace = True)
    if binarize==True:
        results.loc[results['prediction_percentage'] >= decision_prob, 'prediction'] = 1
        results.loc[results['prediction_percentage'] < decision_prob, 'prediction'] = 0
    else:
        results.loc[results['pollock_prob'] >= decision_prob, 'prediction'] = 1
        results.loc[results['pollock_prob'] < decision_prob, 'prediction'] = 0
    results.loc[results['actual'] == results['prediction'], 'failed'] = False
    results.loc[results['actual'] != results['prediction'], 'failed'] = True 
    return results

def do_all_voting(results, binarize_prob=0.5, decision_prob=0.5,simplify = False,failed = False):
    # full_results = vote_system(results, binarize_prob=binarize_prob, decision_prob=decision_prob)[['painting','actual']]
    options = [True,False]
    first = True
    for gov_type in options:
        gov_str = 'sizes'+str(gov_type)[0]
        for vote_type in options:
            r = vote_system(results, one_vote=gov_type, binarize=vote_type, binarize_prob=binarize_prob, decision_prob=decision_prob)
            if first:
                full_results = r[['painting','actual']].copy()
                first = False
            run_type = gov_str+'_binary'+str(vote_type)[0]
            full_results[run_type +'_prediction'] = r['prediction']
            full_results[run_type +'_failed'] = r['failed']
        full_results[gov_str +'_predict_percentage'] = r['prediction_percentage']
        full_results[gov_str +'_pollock_prob'] = r['pollock_prob']
    
        simp_full_results = full_results[['painting'] + [column for column in full_results.columns if column.endswith('failed')]]
    if simplify:
        full_results = simp_full_results
    if failed:
        full_results = full_results[simp_full_results.drop(columns=['painting']).any(axis=1)]
    return full_results


def add_new_painting_to_master(master,dims,
                               catalog = '',
                               title ='',
                               year ='',
                               artifacts ='',
                               source = '',
                               dimensions_cm = '0x0',
                               medium = '',
                               painting = '',
                               save=False,
                               row=0
                              ):
    #dims should be a single line df, code takes first line either way
    
    dims['width_px']=dims.apply(lambda row: row['dims(w,h)'][0],axis=1)
    dims['height_px']=dims.apply(lambda row: row['dims(w,h)'][1],axis=1)
    dims['file'] = dims.apply(lambda row: row['file'].split('_crop')[0],axis=1)
    file = dims.file.iloc[row]
    height_cm = float(dimensions_cm.split('x')[0])
    width_cm = float(dimensions_cm.split('x')[1])
    width_px =dims.width_px.iloc[row]
    height_px = dims.height_px.iloc[row]
    d = {'file':[file],
    'catalog':[catalog],
    'title':[title],
    'year':[year],
    'artifacts':[artifacts],
    'source':[source],
    'dimensions_cm':[dimensions_cm],
    'height_cm':[height_cm],
    'width_cm':[width_cm],
    'width_px':[width_px],
    'height_px':[height_px],
    'px_per_cm_width':[width_px/width_cm],
    'px_per_cm_height':[height_px/height_cm],
    'medium':[medium],
    'painting':[painting]}
    df=pd.DataFrame(data=d)
    master=master.append(df)
    if save:
        master.to_parquet(save + '.parquet')
    return master

def make_dims_after_external_cropped(image_path, save = False,contain = (10000,10000),dim_name = 'dims.parquet'):
    d = []
    image_path = Path(image_path)
    images = os_listdir(image_path)
    raw_path = Path(image_path.parent,'Processed/Raw/Full')
    if not os.path.exists(raw_path):
        os.mkdir(raw_path.parent.parent)
        os.mkdir(raw_path.parent)
        os.mkdir(raw_path)    
    for image in images:
        painting = Image.open(Path(image_path,image))
        # painting = imageio.imread(Path(image_path,image))
        if contain and (painting.width > contain[0] or painting.height > contain[1]):
            painting = ImageOps.contain(painting, contain)
        painting.save(Path(raw_path,Path(image).stem + '_cropped.tif'))
        # imageio.imwrite(Path(raw_path,Path(image).stem + '_cropped.tif'),painting)
        # height=len(painting)
        # width=len(painting[0])
        height = painting.height
        width = painting.width
        d.append([width,height])
    images = [file.split('.')[0] for file in images]
    data = {'file':images,'savePath':[str(raw_path)]*len(images),'dims(w,h)': d}
    dims = pd.DataFrame(data=data)
    if save:
        dims.to_parquet(Path(Path(save).parent,dim_name))
    return dims 

def get_painting_name_with_most_slices(df, name_col='painting_name'):
    num_slices = df.groupby(name_col).count().max(axis=1)
    return num_slices.idxmax()


def pad_row_col_str(pos, num_digits):
    ## add zeros such that all R## and C## are the same length
    return pos[0] + pos[1:].rjust(num_digits, '0')


def add_column_row_position_to_df(df):
    # add a column which just has C## and R## to df 
    tmp = df.copy()
    tmp['column_row'] = df['file'].apply(lambda x: x.split('_descreened_')[-1][:-(len('.tif'))])
    return tmp


def find_longest_position_str(df, position_col='column_row'):
    # Finds the length of the longest '##' substring in all 'C##_R##' strings
    assert position_col in df.columns, 'Add position_col to df'
    
    longest = df['column_row'].str.len().max()
    longest -= 1 # remove '_'
    position_length = np.ceil(longest/2) 
    # divide by 2 to get length of C## and R##, round up in case one is different than the other
    
    num_digits = int(position_length - 1)
    return num_digits
    

def stretch_col_row(col_row, num_digits):
    col, row = col_row.split('_')
    col = pad_row_col_str(col, num_digits)
    row = pad_row_col_str(row, num_digits)
    return col + '_' + row


def stretch_to_neg_one(df, col_name='pollock_prob'):
    df[col_name] = 2 * df[col_name] - 1
    return df


def make_feature_samples_for_all_slices(df, name_str='file',  pos_col='column_row', split_str='_decreened_', pic_ext='.tif', prob_col='pollock_prob'):
    '''
    df should be a results dataframe
    '''
    tmp = df.copy()
    
    tmp[pos_col] = tmp[name_str].apply(lambda x: x.split(split_str)[-1][:-(len(pic_ext))])
    num_digs = find_longest_position_str(tmp, position_col=pos_col)
    tmp[pos_col] = tmp[pos_col].apply(lambda x: stretch_col_row(x, num_digs))
    
    tmp = stretch_to_neg_one(tmp, col_name=prob_col)
    tmp[pos_col] = tmp['slice_size'].astype(str) + '_' + tmp[pos_col]

    feature_samples = tmp.pivot(columns='painting_name', index=pos_col, values=prob_col)
    feature_samples = feature_samples.fillna(0)

    # feature is now a N x Samples shaped object. Can be used with scikit-learn -- N is the number of features
    return feature_samples


def make_feature_samples_by_slice_size(df, painting_col='painting', group_col='slice_size', prob_col='pollock_prob'):
    '''
    df should be a results dataframe 
    '''
    painting_df = df.groupby([painting_col, group_col]).mean()
    painting_df = painting_df.reset_index()
    painting_df = stretch_to_neg_one(painting_df, col_name=prob_col)

    piv_df = painting_df.pivot(index=painting_col, columns=group_col, values=prob_col)
    # piv_df = stretch_to_neg_one(piv_df, col_name=prob_col)
    piv_df.fillna(0, inplace=True)
    
    return piv_df


def make_category_for_each_sample(df, feature_df, sample_col='painting_name'):
    category = df.groupby(sample_col).all()*1
    category = category.loc[feature_df.T.columns, :]
    category = category.loc[:, 'actual']
    return category


def make_pollock_prob_hist_by_group(df):
    fig, ax = plt.subplots(2, 4, constrained_layout=True)
    upper_lim = 1
    lower_lim = 0
    step_size = 0.1
    bins = np.arange(lower_lim, upper_lim + step_size, step_size)
    bins

    groups = df['group'].unique()
    n_groups = len(groups)

    loc_dict = {
        'A':[0,0],
        'C':[0,1],
        'D':[0,2],
        'E':[1,0],
        'F':[1,1],
        'G':[1,2],
        'J':[0,-1],
        'P':[1, -1]
    }


    for group in groups:
        x,y = loc_dict[group]
        ax[x,y].hist(df[df['group']==group]['pollock_prob'], bins=bins, density=True)
        ax[x,y].set_title(group)
        # ax[x,y].set_xticks(bins)

    plt.show()
    
def split_dataframe(df, chunk_size = 10): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    if not len(chunks[-1]):
        chunks = chunks[:-1]
    return chunks

def flip_image_dir(directory_a,directory_b,match_str = "*.tif",save_ending = '.tif',append = '_flip_v'):
    input_files = glob.glob(os.path.join(directory_a, match_str))
    # print(input_files)
    for file in input_files:
        img = cv2.imread(file)
        horizontal_img = cv2.flip(img, 1)
        write_path = os.path.join(directory_b, os.path.basename(file).split(".")[0] + append + save_ending)
        # print(write_path)
        cv2.imwrite(write_path,
                horizontal_img)
        
def flip_images(pathA,pathB,match_str = "*.tif"):
    # pathA = 'Paintings/Processed/Descreened'
    # pathB = 'Paintings_flipped/Processed/Descreened'
    foldersA = get_folders(pathA)
    # foldersB = get_folders(pathB)

    for folder in foldersA:
        # print(folder)
        folder_management(folder,pre_path = pathB,append_date = False,overwrite = True)
        flip_image_dir(os.path.join(pathA,folder), os.path.join(pathB,folder),match_str=match_str)
        
def get_paths(base_folder,file_list):
    paths = []
    for file in file_list:
        paths.append(os.path.join(base_folder,file))
    return paths


def load_metrics(path):
    folders_conv = ['group_accuracies','incorrect','folders','valid_set','hold_out_catalog','hold_out_files','hold_out_duplicates']
    conversion = {item:literal_eval for item in folders_conv}
    return pd.read_csv(path, converters=conversion,low_memory=False)

def get_paintings_list(metrics,paintings = 'valid_set'):
    if isinstance(paintings,str):
        if paintings in metrics.columns.tolist():
            paintings = metrics[paintings].iloc[0]
        else:
            paintings = [paintings]
    elif isinstance(paintings,list):
        paintings = paintings
    else:
        print("Warning:unsopported 'paintings' input")
    return paintings

def get_model_accuracy(results):
    r = vote_system(results,one_vote=False, binarize=False, binarize_prob=0.5, decision_prob=0.50)
    return r.apply(lambda row:row.pollock_prob if row.actual else 1-row.pollock_prob,axis = 1).mean()

def get_bool(prompt):
    while True:
        try:
           return {"true":True,"false":False}[input(prompt).lower()]
        except KeyError:
           print("Invalid input please enter True or False!")
        

def look_up_P_to_J(painting_name, file_name='mapP-J.csv'):
    if painting_name[0] == 'P':
        col = 'file_P'
    elif painting_name[0] == 'J':
        col = 'file_J'
    else:
        error('must be a J of P painting')
    
    df = pd.read_csv('mapP-J.csv')
    
    df = df[df[col].notna()]
    df = df[df[col].str.startswith(painting_name)]
    return df



def get_results_for_all_painting_images(results_df, painting_name, col='painting_name'):
    
    assert col in results_df.columns, f'{col} not in results_df, assign column via "col=" in  function'
    
    map_df = look_up_P_to_J(painting_name, file_name='mapP-J.csv')
    p_names = list(map_df['file_P'].unique())
    j_names = list(map_df['file_J'].unique())
    
    combined_names = p_names + j_names
    
    return results_df[results_df[col].isin(combined_names)]

def p_jpcr_diff(results,vote=False,binary=False,binprop=.5,decprop=.5):
    results=vote_system(results, one_vote=vote, binarize=binary, binarize_prob=binprop, decision_prob=decprop)
    results = results.rename(columns={'painting': 'painting_name'})
    paint_res = results.groupby('painting_name').mean()
    paint_res.reset_index(inplace=True)
    temp=paint_res[(paint_res['painting_name'].str.startswith('J')) | (paint_res['painting_name'].str.startswith('P'))].sort_values('pollock_prob',     ascending=True)
    if len(temp) >0:
        df=get_results_for_all_painting_images(paint_res, temp['painting_name'].iloc[0])
        pline=df[(df['painting_name'].str.startswith('P'))]
        jline=df[(df['painting_name'].str.startswith('J'))]
        df = pd.merge(pline, jline, on=["actual", "actual"])
        emptydf = pd.DataFrame(columns=df.columns)
        test=paint_res[(paint_res['painting_name'].str.startswith('P'))]
        testlist=test['painting_name'].to_list()
        for i in testlist:
            df=get_results_for_all_painting_images(paint_res, i)
            pline=df[(df['painting_name'].str.startswith('P'))]
            jline=df[(df['painting_name'].str.startswith('J'))]
            df = pd.merge(pline, jline, on=["actual", "actual"])
            emptydf = pd.concat([emptydf, df], ignore_index=True, sort=False)
        emptydf['pred_diff']=(emptydf['prediction_percentage_x']-emptydf['prediction_percentage_y']).abs()
        emptydf['prob_diff']=(emptydf['pollock_prob_x']-emptydf['pollock_prob_y']).abs()
        emptydf = emptydf.rename(columns={'painting_name_x': 'p_Name', 'prediction_percentage_x': 'p_prediction_percentage', 'pollock_prob_x': 'p_pollock_prob',
         'failed_x': 'p_failed' ,
        'painting_name_y': 'jpcr_Name', 'prediction_percentage_y': 'jpcr_prediction_percentage', 'pollock_prob_y': 'jpcr_pollock_prob',
          'failed_y': 'jpcr_failed'})
        emptydf=emptydf.drop(['actual','prediction_x','prediction_y'], axis=1)
        emptydf=emptydf.sort_values(by='prob_diff', ascending=False)
        tempdf=emptydf.copy()
        emptydf.loc['std'] = tempdf.std()
        emptydf.loc['mean'] = tempdf.mean()
    else:
        nan = float("nan")
        d = {'p_Name': {'std': nan, 'mean': nan},
             'p_prediction_percentage': {'std': nan, 'mean': nan},
             'p_pollock_prob': {'std': nan, 'mean': nan},
             'jpcr_Name': {'std': nan, 'mean': nan},
             'jpcr_prediction_percentage': {'std': nan, 'mean': nan},
             'jpcr_pollock_prob': {'std': nan, 'mean': nan},
             'pred_diff': {'std': nan, 'mean': nan},
             'prob_diff': {'std': nan, 'mean': nan}}
        emptydf = pd.DataFrame(data = d)
    return emptydf


def test_folders(master,
                 base_folder = 'Paintings/Processed/Raw/',
                 full_paint_folder = 'Full',
                 output_sizes_cm=False,
                 from_master = True,
                 verbose = False,
                 files_from_dir = False,
                 count = False):
    inconsistent = []
    expected = []
    in_folder = []
    folder_size = []
    names = []
    if files_from_dir:
        samples = AbstractArtData(root_dir=os.path.join(base_folder,full_paint_folder))
        file_names = [item.split('_c')[0] for item in samples.image_names if not item.startswith('.')]
    else:
        file_names = master.file.tolist()
    if not output_sizes_cm:
        output_sizes_cm = sorted([int(item) for item in os.listdir(base_folder) if item.isnumeric()])
    for file_name in file_names:
        if verbose:
            print('checking consistency for ', file_name)
        if master['file'].isin([file_name]).any():
            resolution = float(master.loc[master['file'] == file_name]['px_per_cm_height'])
            for output_size_cm in output_sizes_cm:
                output_size_px = round(output_size_cm * resolution)
                if from_master:
                    expect_num_slices = count_tile_from_master(master, file_name,output_size_px)
                else:
                    expect_num_slices = count_tile_from_image(sample,output_size_px)
                slices = os.listdir(os.path.join(base_folder,str(output_size_cm)))
                folder_num_slices = len([item for item in slices if item.startswith(file_name + '_c')])
                expected.append(expect_num_slices)
                in_folder.append(folder_num_slices)
                folder_size.append(output_size_cm)
                names.append(file_name)
                if expect_num_slices != folder_num_slices:
                    # expected.append(expect_num_slices)
                    # in_folder.append(folder_num_slices)
                    # folder_size.append(output_size_cm)
                    if output_size_cm < min(master[master.file == file_name].height_cm.iloc[0],master[master.file == file_name].width_cm.iloc[0]):
                        inconsistent.append({'file':file_name,'expected':expect_num_slices,'in_folder':folder_num_slices,'folder_size':output_size_cm})
                        if verbose:
                            print('expected',expect_num_slices,'folder',folder_num_slices,' for', file_name, 'in',output_size_cm)
        # inconsistent.append({'file':file_name,'expected':expected,'in_folder':in_folder,'folder_size':folder_size})
        else:
            print('...master does not contain file_name ', file_name)
    if count:
        return pd.DataFrame({'file':names,'tile_size':folder_size,'expected':expected,'in_folder':in_folder})
    return inconsistent

def count_tile_from_master(master, file_name,output_size,cm = False):
    if str(output_size).isnumeric():
        if cm:
            output_size = round(float(output_size)*float(master[master.file == file_name].px_per_cm_height.iloc[0]))      
    elif str(output_size) == 'Max':
        output_size = min(int(master[master.file == file_name].height_px.iloc[0]),int(master[master.file == file_name].width_px.iloc[0])) 
    else:
        print('unidentified output size type')
    output_size = SliceCrop(int(output_size),0).output_size #throw in to SliceCrop class to get output_size formatted
    h = master[master.file == file_name].height_px.iloc[0]
    w = master[master.file == file_name].width_px.iloc[0]
    px_dims = [h,w]
    num_slices =[int(px_dims[i]/output_size[i]) for i in (0,1)]   
    
    return num_slices[0]*num_slices[1]
def get_tile_counts(master,
                  base_folder = 'Paintings/Processed/Raw/',
                  full_paint_folder = 'Full',
                  output_sizes_cm=range(10,365,5),
                  from_master = True,
                  verbose = False,
                  files_from_dir = False,
                  save = False):
    counts = test_folders(master,
                  base_folder = base_folder,
                  full_paint_folder = full_paint_folder,
                  output_sizes_cm=output_sizes_cm,
                  from_master = from_master,
                  verbose = verbose,
                  files_from_dir = files_from_dir,
                  count = True)
    counts = pd.merge(left = counts, right = master[['file','set','artist']], on = 'file')
    counts = counts[(counts.expected != 0) & (counts.in_folder != 0)]
    if save:
        counts.to_csv(save)
    return counts
def count_tile_from_image(sample, output_size):
    stem = Path(sample['name']).stem
    output_size = SliceCrop(int(output_size),0).output_size #throw in to SliceCrop class to get output_size formatted
    num_slices = [int(sample['image'].shape[i]/output_size[i]) for i in (0,1)]    
    
    return num_slices[0]*num_slices[1]

def get_rows_columns_from_files(files):
    columns = [int(item.split('_C')[1].split('_')[0]) for item in files]
    num_columns = max(columns)+1
    rows = [int(item.split('_R')[1].split('.')[0]) for item in files]
    num_rows= max(rows)+1
    return num_rows,num_columns,rows,columns

def get_accuracy_column(df_with_actual_and_pollock_prob,sort = False):
    df_with_actual_and_pollock_prob['accuracy']=df_with_actual_and_pollock_prob.apply(lambda row: np.abs((1-row.actual)-row.pollock_prob),axis=1)
    if sort:
        df_with_actual_and_pollock_prob = df_with_actual_and_pollock_prob.sort_values(sort)
    return df_with_actual_and_pollock_prob

def get_slice_sizes(results,num=4):
    folders_no_max = sorted(list(set(results.slice_size.tolist())))[:-1]
    folders_no_max = sorted([int(item) for item in folders_no_max])
    if num == 4:
        spacing = int(round((len(folders_no_max)+1)/(num-1)))
        slice_sizes = [str(folders_no_max[0]),str(folders_no_max[spacing]),str(folders_no_max[2*spacing]),'Max']
    else:
        spacing = int(round((len(folders_no_max))/(num-1)))
        slice_sizes = [str(folders_no_max[i*spacing]) for i in range(num-1)] + ['Max']  
    return slice_sizes

def get_sequences(rows):
    my_sorted_list = rows
    my_sequences = []
    for idx,item in enumerate(my_sorted_list):
         if not idx or item-1 != my_sequences[-1][-1]:
             my_sequences.append([item])
         else:
             my_sequences[-1].append(item)
    max(my_sequences, key=len)

    return my_sequences

def get_num_slices_in_painiting(M,file,sizes):
    slices = {}
    for i in sizes:
        slices[str(i)] = count_tile_from_master(M, file,i,cm=True)
    return sum(slices.values())

def get_summary_composit(folder,first_row = False, save_path = ''):
    summaries = []
    for path in [ii for ii in os.listdir(folder) if not ii.startswith('.')]:
        if first_row:
            summaries.append(pd.read_csv(Path('summaries',path)).iloc[[0]])
        else:
            summaries.append(pd.read_csv(Path('summaries',path)))
    summary_composit = pd.concat(summaries)
    summary_composit.drop(columns = ['Unnamed: 0'],axis = 1,inplace=True)
    summary_composit.reset_index(inplace = True)
    summary_composit.drop(columns = 'index',axis=1,inplace=True)
    summary_composit.to_csv(Path(save_path,'summary_composit.csv'))
    return summary_composit

#just makes a full composit from 
def get_summary_full_composite(folder,save_path = ''):
    summary = []
    summary.append(get_summary_composit(folder,first_row=True))
    summary.append(get_summary_composit(folder))
    summary = pd.concat(summary)
    summary.to_csv(Path(save_path,'summary_composit.csv'))
    return summary

def ldir(path):
    return [item for item in os.listdir(path) if not item.startswith('.')]

def get_max_output_size_px_based(master,file_name):
    width_cm = (master[master.file == file_name].width_px/master[master.file == file_name].px_per_cm_height).iloc[0]
    height_cm = (master[master.file == file_name].height_px/master[master.file == file_name].px_per_cm_height).iloc[0]
    return int(min(width_cm,height_cm))

def get_max_size_from_results(results,file_name):
    return np.max(pd.to_numeric(results[results.file.str.startswith(file_name + '_cropped')].slice_size,errors = 'coerce').dropna().astype('int'))

def tile_images(master,read_dir,center_tile = True,update = False,start = 5,end = False,add=0,verbose = True):
    M = master[master['px_per_cm_height'].notnull()]
    # read_dir = 'Bristol-Overlapping/Processed/Raw/Full/'
    if not end:
        end = int(min(max(M.height_cm),max(M.width_cm)) + 5.0)
    # else:
    #     assert(isinstance(end,int),'end needs to be an integer (multiple of 5)')
    output_sizes_cm = list(range(start,end,5))
    output_sizes_cm.reverse()
    if not center_tile:
        if add > 0:
            overlap_path = str(Path(read_dir).parents[0])+'-overlap'+'-'+str(add)
        else:
            overlap_path = str(Path(read_dir).parents[0])+'-overlap'
        if not os.path.exists(overlap_path):
            os.mkdir(overlap_path) 
    
    for output_size_cm in output_sizes_cm:
        # t = time.time()
        print_if(verbose,output_size_cm)
        tile_images_from_dir(read_dir,M, output_size_cm,update = update,center_tile = center_tile,add = add,verbose = verbose)
        # elapsed = time.time() - t
        # print(elapsed)
    print_if(verbose,'Max')
    tile_images_from_dir_max_crop(read_dir,M,update = update,write_folder = 'Max',center_tile = center_tile,add=add,verbose = verbose)

def convert_to_BW(file_name,read_path = 'Paintings/Processed/Raw/Full',write_path = 'BW_converted/Processed/Raw/Full',ending = '_cropped.tif',add_tag = False):
    # Load the image
    full_read_path = str(Path(read_path,file_name+ending))
    if add_tag:
        full_write_path = str(Path(write_path,file_name+add_tag+ending))
    else:
        full_write_path = str(Path(write_path,file_name+ending))
    image = cv2.imread(full_read_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(full_write_path, bgr_image)

def print_if(condition, *args, **kwargs):
    if condition:
        print(*args, **kwargs)

def unique_len(df,column = 'painting'):
    return len(set(df[column]))

def get_resolution_df(master,factors = [1]+list(range(2,22,2))+[50,100],save = False):
    df = master.copy()
    files = df.file.tolist()
    # Initialize an empty DataFrame to store the repeated rows
    repeated_df = pd.DataFrame()

    # Repeat and divide columns
    for file_name in files:
        for i in factors:
            repeated_row = df[df.file == file_name].copy()  # Copy the original row
            repeated_row[['width_px', 'height_px','px_per_cm_width','px_per_cm_height']] /= i  # Divide selected columns by the fraction
            repeated_row['file'] = file_name +'_1_divided_by_' + str(i)
            repeated_df = pd.concat([repeated_df, repeated_row])  # Append the repeated row to the DataFrame

    # Reset the index of the repeated DataFrame
    repeated_df = repeated_df.reset_index(drop=True)
    if save:
        repeated_df.to_parquet(save + '.parquet')
    return repeated_df

def os_listdir(path, ignore_hidden = True):
    file_paths = os.listdir(path)
    if ignore_hidden:
        files = [file for file in file_paths if not file.startswith('.')]
    return files

def list_from_df_or_list(df_or_list,column = 'file'):
    if isinstance(df_or_list, pd.DataFrame): #checks if you inputed master instead
        files = df_or_list[column].tolist()
    else:
        files = df_or_list
    assert isinstance(files,list) , 'input type not master data frome with file column or list'
    return files

def make_resolution_images(files, 
                           img_path_testing = 'Paintings/Processed/Raw/Full',
                           save_path = 'resolution/Processed/Raw/Full',
                           factors = [1]+list(range(2,22,2))+[50,100],
                           selected = False,
                           interpolation=cv2.INTER_AREA):
    files = list_from_df_or_list(files)
    if selected:
        if isinstance(selected,str):
            files = [selected]
        else:
            files = selected
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in files:
        img_full = cv2.imread(os.path.join(img_path_testing,file + '_cropped.tif'))
        for factor in factors:
            fraction = 1/factor
            img = cv2.resize(img_full, (0,0), fx = fraction, fy = fraction, interpolation=interpolation)
            cv2.imwrite(os.path.join(save_path,file +'_1_divided_by_' + str(factor) + '_cropped.tif'), img)

def get_base_file_name(df_or_list,column = 'file',split_on = '_c',split_element = 0):
    df_or_list = list_from_df_or_list(df_or_list,column = column)
    files = sorted(list(set([item.split(split_on)[split_element] for item in df_or_list])))
    return files

def startswith(df,test_column = 'file',set_column = 'pollock',strings = ('P','J')):
    condition = df[test_column].str.startswith(('P','J'))
    df[set_column] = condition
    return df

def combine_list_of_arrays(combined_list):
    array_stack = np.stack(combined_list)
    average_array = np.mean(array_stack, axis=0)
    return average_array

def round_special_no_zero_neg(num,decimal=1):
    rounded_num = round(num, decimal)
    if math.copysign(1, num) == -1 and rounded_num == 0:
        rounded_num = math.ceil(num)
    rounded_num = "{:.{}f}".format(rounded_num, decimal)
    
    return rounded_num

def get_row_values(df, columns):
    row = df.iloc[0][columns].values.tolist()
    return row

def square_crop(image):
    # im = Image.open(<your image>)
    width, height = image.size   # Get dimensions
    new_width = min([width,height])
    new_height = min([width,height])

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = image.crop((left, top, right, bottom))

    width, height = im.size 
    if width != height:
        if (width - height) == 1:
            new_width = width - 1
            im = im.crop((0, 0, new_width, height))
        elif (height - width) == 1:
            new_height = height -1
            im = im.crop((0, 0, width, new_height))
    width, height = im.size
    assert(height==width)


    return im

def copy_pollock_maps(source_dir='viz/individual/',save_dir = '',add = 0,paintings = ['H2','P2(V)', 'P43(W)'],painting_names = ['gray.png','overlay.png','pollock_map.png']): 
    for i,painting in enumerate(paintings):
        for file in painting_names:
            folder_str = painting +'_overlap'+'_'+str(add)
            file_str = 'select_' + str(i) +'_'+file
            source_file = os.path.join(source_dir, folder_str,folder_str,file)  
            assert os.path.exists(source_file) ,"Doesn't look like the path exists. Maybe set 'do_viz' and/or 'do_comparison_viz' to 'True' first?"
            destination_file = os.path.join(save_dir,file_str)
            shutil.copyfile(source_file, destination_file)
        print(f'pollock map files for {painting} copied from {source_dir} to {save_dir}')

def compare_BW_converted(results1,results2,results1_name = 'C',results2_name = 'BW'):
    r1 = vote_system(results1)
    r2 = vote_system(results2)
    r = pd.merge(left = r1[['painting','pollock_prob']],right = r2[['painting','pollock_prob']], on = 'painting',suffixes=('_' + results1_name,'_' + results2_name))
    r['PMF_diff'] = r['pollock_prob_' + results1_name]-r['pollock_prob_' + results2_name]
    r['abs_PMF_diff'] = np.abs(r['pollock_prob_' + results1_name]-r['pollock_prob_' + results2_name])
    r = r.sort_values('abs_PMF_diff',ascending = False)
    return r

def remove_background_and_save(input_image_path, output_png_path,thresh = 120):
    # Read the input image using PIL
    sig = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale for easier processing
    sig_gray = cv2.cvtColor(sig, cv2.COLOR_BGR2GRAY)

    # Threshold the image to separate the signature from the background
    ret, alpha_mask = cv2.threshold(sig_gray, thresh, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite(output_png_path, alpha_mask)

    blue_mask = sig.copy()
    blue_mask[:, :] = (255, 0, 0)

    sig_color = cv2.addWeighted(sig, 1, blue_mask, 0.5, 0)

    b, g, r = cv2.split(sig_color)

    # Create a list of the four arrays with the alpha channel as the 4th member. These are four separate 2D arrays.
    new = [b, g, r, alpha_mask]

    # Use the merge() function to create a single, multi-channel array.
    png = cv2.merge(new, 4)

    # Save the transparent signature a PNG file to retain the alpha channel.
    cv2.imwrite(output_png_path, png)

    return png

def str_round(number,round_to):
    return f"{round(number,round_to):.{round_to}f}"

def append_modified_row(df,row,new_col_values = {'file':'new_file_str','title':'new_title_str'}):
    # Extract the first row
    # first_row = df.iloc[0].copy()
    Row = row.copy()

    # Modify the values of 'file' and 'title' columns
    for key in new_col_values:
        Row[key] = new_col_values[key]

    # Append the modified row back to the DataFrame
    df = df.append(Row, ignore_index=True)
    return df
def tile_res_specific(title,master_testing,
                    output_size_cm = 20,
                    res_factor = 30,
                    img_path_testing = 'Processed/Raw',
                    verbose = False,
                    center_tile = True,
                    add = 0,
                    processed_sub_folder = 'Res',
                    update = False
                    ):
        res_path = os.path.join(Path(img_path_testing).parent,processed_sub_folder,'Full')
        make_resolution_images(master_testing,img_path_testing = os.path.join(img_path_testing,'Full'),save_path = res_path,factors = [res_factor],selected = title)
        selected_res_title = [file for file in os_listdir(res_path) if file.startswith(title+'_1_divided_by_' + str(res_factor))]
        master_res = get_resolution_df(master_testing,factors = [res_factor],save = False)
        tile_images_from_dir(res_path,master_res, output_size_cm,update = False,center_tile = center_tile,add = add,verbose = verbose,selected = selected_res_title)

def custom_squish_mapping(value, center_value, squish_factor):
    scaled_value = center_value - (center_value - value) * squish_factor
    return scaled_value

def check_file_in_subfolders(root_directory, target_file):
    """
    Check if a file exists in any subfolder under the given root directory.

    Parameters:
    root_directory (str): The root directory to start searching from.
    target_file (str): The file name to search for.

    Returns:
    bool: True if the file is found in any subfolder, False otherwise.
    """
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if target_file in filenames:
            return True
    return False
def select_rows(dataframe,train_num = 38, valid_num = 9,hold_out_num = 3,groups = ['A', 'C', 'F', 'G', 'J', 'P']):
    valid_rows = dataframe[dataframe['set'] == 'valid'].sample(valid_num)
    train_rows = dataframe[dataframe['set'] == 'train'].sample(train_num)
    hold_out_rows = dataframe[dataframe['set'] == 'hold_out'].sample(hold_out_num)

    # Ensure at least one row of each group in the selected rows
    for group in groups:
        if group not in valid_rows['group'].values:
            valid_rows = pd.concat([valid_rows, dataframe[(dataframe['set'] == 'valid') & (dataframe['group'] == group)].sample(1)])

        if group not in train_rows['group'].values:
            train_rows = pd.concat([train_rows, dataframe[(dataframe['set'] == 'train') & (dataframe['group'] == group)].sample(1)])

        if group not in hold_out_rows['group'].values:
            hold_out_rows = pd.concat([hold_out_rows, dataframe[(dataframe['set'] == 'hold_out') & (dataframe['group'] == group)].sample(1)])

    # Combine the selected rows
    selected_rows = pd.concat([valid_rows, train_rows,hold_out_rows])

    return selected_rows