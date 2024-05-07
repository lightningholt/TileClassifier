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
def tile_an_image(sample, output_size, center_tile = True ,rescale = True,write_dir = None,image_size = 256):
    stem = Path(sample['name']).stem
    # print(stem)
    # print(output_size)
    output_size = SliceCrop(int(output_size),0).output_size #throw in to SliceCrop class to get output_size formatted
    
    if center_tile:
        start_loc = get_tile_start(sample,output_size)
    else:
        start_loc = (0,0)
        
    sc = SliceCrop(output_size,start_loc)
    num_slices = [int(sample['image'].shape[i]/output_size[i]) for i in (0,1)]
    # slice_locs = []
    # slices = []
    scaled_slices = []
    
    if rescale:
        composed = transforms.Compose([sc,Rescale(image_size)])
    else:
        composed = transforms.Compose([sc])    
        
    if write_dir:
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
            
    
    # slice_loc = start_loc
    for i in range(num_slices[0]):
        for j in range(num_slices[1]):
            slice_loc = (start_loc[0] + output_size[0]*i,start_loc[1] + output_size[1]*j)
            # slice_locs.append(slice_loc)
            sc.borders = slice_loc
            slc = composed(sample)
            scaled_slices.append(slc)
            if write_dir:
                location_tag ='_C'+str(i).zfill(len(str(num_slices[0])))+'_R'+str(j).zfill(len(str(num_slices[1])))
                full_path = os.path.join(write_dir,stem+location_tag+'.tif')
                imageio.imwrite(full_path,slc['image'])
                
            
            
    # if rescale:
    #     scale = Rescale(256)
    #     scaled_slices = []
    #     for slice in slices:
    #         scaled_slices.append(scale(slice))
            
    # if write_dir:  # How do we want to write these? Particularily the naming convention.
    
    
        
        
    
    return scaled_slices

def get_tile_start(sample,output_size):
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

def get_fitting_samples(master, samples, output_size_cm):
    Mb = filter_to_big_enough_images(master, output_size_cm)
    # print(Mb)
    fitting_items = tuple([item +'_cropped' for item in Mb.file.tolist()])
    # print('fitting_items=',fitting_items)
    fitting_files = [sample for sample in samples.image_names if sample.startswith(fitting_items)]
    # print('image_names=',samples.image_names)
    print(fitting_files)
    return fitting_files

def tile_images_from_dir(read_dir, master, output_size_cm, redo = False, selected = None,update = True, image_size = 256):
    #General use: Does not overwrite files in directories
    #Ex. tile_images_from_dir('Paintings/Processed/Raw/Full/',M, output_size_cm)
    
    #Overwrite files:
    #Ex. tile_images_from_dir('Paintings/Processed/Raw/Full/',M, output_size_cm, update = False)
    
    #Run for selected files: Does overwrite files
    #Ex. tile_images_from_dir('Paintings/Processed/Raw/Full/',M, output_size_cm,selected =selected_tifs)
    failed_to_tile = []
    
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
    samples.image_names = get_fitting_samples(master, samples, output_size_cm)
    # print(samples.image_names)
    
    for sample in samples:     
        # file_name = sample
        file_name = sample['name'].split('_cropped')[0]
        # print('checking', file_name)
        if file_name in master['file'].unique():
        # print(master.loc[master['file'] == file_name]['px_per_cm_height'])
            # print(file_name)
            resolution = float(master.loc[master['file'] == file_name]['px_per_cm_height'])
            print(file_name, ' fits at least one box inside it')
            output_size_px = round(output_size_cm * resolution)
            tile_an_image(sample,output_size_px,write_dir=write_dir, image_size=image_size)
            # else:
            #     print(file_name, 'was skipped because box bigger than a dimension')
        else:
            failed_to_tile.append(file_name)
            # print('Failed to tile: ',file_name, ' at crop size: ', output_size_cm)
    if len(failed_to_tile)>0:
        print('Failed to tile: ',failed_to_tile, ' at crop size: ', output_size_cm)
            
def get_dir_core_differences(read_dir,write_dir,split_str='_cropped',remove_hidden = True):
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
    

def create_onepainting(results,file_start,slice_size):
    one_painting = results[results.file.str.startswith(file_start)]
    one_painting=one_painting[one_painting.slice_size==slice_size]
    column=one_painting.apply(lambda row: int(row.file.split('_C')[1].split('_')[0]),axis=1)
    row=one_painting.apply(lambda row: int(row.file.split('_R')[1].split('.')[0]),axis=1)
    one_painting['C']=column
    one_painting['R']=row
    column2=one_painting.apply(lambda row: ('C'+row.file.split('_C')[1].split('_')[0]),axis=1)
    row2=one_painting.apply(lambda row: ('R'+row.file.split('_R')[1].split('.')[0]),axis=1)
    one_painting['C2']=column2
    one_painting['R2']=row2
    
    return one_painting    


def green(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray = gray.astype(np.float32)

    # create blue image
    green  = np.full_like(image, (0,255,0), np.float32) / 255

    # multiply gray by blue image
    result = cv2.multiply(gray, green)
    

    return result

def concat_images(image_set, how, with_plot=False):
    # dimension of each matrix in image_set
    shape_vals = [imat.shape for imat in image_set]

    # length of dimension of each matrix in image_set
    shape_lens = [len(ishp) for ishp in shape_vals]

    # if all the images in image_set are read in same mode
    channel_flag = True if len(set(shape_lens)) == 1 else False

    if channel_flag:
        ideal_shape = max(shape_vals)
        images_resized = [
            # function call to resize the image
            resize_image(image_matrix=imat, nh=ideal_shape[0], nw=ideal_shape[1]) 
            if imat.shape != ideal_shape else imat for imat in image_set
        ]
    else:
        return False

    images_resized = tuple(images_resized)

    if (how == 'vertical') or (how == 0):
        axis_val = 0
    elif (how == 'horizontal') or (how == 1):
        axis_val = 1
    else:
        axis_val = 1

    # numpy code to concatenate the image matrices
    # concatenation is done based on axis value
    concats = np.concatenate(images_resized, axis=axis_val)

    if with_plot:
        cmap_val = None if len(concats.shape) == 3 else 'gray'
        plt.figure(figsize=(10, 6))
        plt.axis("off")
        plt.imshow(concats, cmap=cmap_val)
        return True
    return concats

def colorize(files,image,one_painting,colors,transparency = [0.5,0.5]):
    
    crow=one_painting[one_painting['file'] == files]
    if crow['prediction'].values[0]==crow['actual'].values[0]:
        file = apply_color(image,colors[0],transparency= transparency)
    else:
        file = apply_color(image,colors[1],transparency= transparency)
    return file

def apply_color(image,color,transparency = [0.5,0.5]):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray = gray.astype(np.float32)

    # create colored image. Two options
    if transparency:
        alpha =  transparency[0]
        beta= transparency[1]
        colored_im  = np.full_like(image, color, np.float32)
        result = cv2.addWeighted(gray, alpha, colored_im,beta,0)
    else:
        colored_im  = np.full_like(image, color, np.float32) / 255
        result = cv2.multiply(gray, colored_im)
    return result

#final_image=image_visualizer(results,'5','F89','F89-t2.jpg',image_base_path = 'Paintings/Processed/Raw',savepath = 'viz',transparency = False)
def image_visualizer(results,slice_size,filename,savename,image_base_path = 'Paintings/Processed/Descreened',colors = [(0,255,0),(0,0,255)],savepath = '.',transparency = [1,0.2]):
    dir_name = Path(image_base_path,slice_size)
    results.slice_size = results.slice_size.astype(str) #converts to str for consitancy
    one_painting=create_onepainting(results,filename,slice_size)
    if one_painting.file.iloc[0].split('cropped_')[1].split('_')[0] == 'descreened' and dir_name.parent.stem == 'Raw':
        print('Warning:descreened results file but pointed to Raw folder. Using images from that raw folder')
        one_painting['file'] = one_painting['file'].str.replace(r'_descreened', '', regex=True)
    elif one_painting.file.iloc[0].split('cropped_')[1].startswith('C') and dir_name.parent.stem == 'Descreened':
        print('Warning:Raw results file but pointed to Descreened folder. Using images from that descreened folder')
        one_painting.file=one_painting.apply(lambda row: row.file.split('_C')[0] + '_descreened_C' + row.file.split('_C')[-1],axis=1)

    vimages=[]
    columns=one_painting.C2.unique().tolist()
    rows=one_painting.R2.unique().tolist()
    counter=0
    clength=len(columns)
    for i in range(clength):
        
        files = [file for file in os.listdir(dir_name) if file.startswith(filename) and (columns[counter]) in file]
        
        images=[]
        
        length = len(files)
        for i in range(length):
            file = os.path.join(dir_name,files[i])
            tempfile=files[i]
            # print(tempfile)
            img = cv2.imread(file)
            img=colorize(tempfile,img,one_painting,colors,transparency = transparency)
            images.append(img)
        himages=concat_images(images, 'horizontal', with_plot=False)
        vimages.append(himages)
        counter+=1
    final_image=concat_images(vimages, 'vertical', with_plot=False)
    if savepath=='none':
        print('not saved')
    else:
        cv2. imwrite(os. path. join(savepath , savename), final_image)
    return final_image
    
def tile_images_from_dir_max_crop(read_dir, master, redo = False, selected = None,update = True,image_size=256,write_folder = 'Max'):
    #General use: Does not overwrite files in directories
    #Ex. tile_images_from_dir('Paintings/Processed/Raw/Full/',M, output_size_cm)
    
    #Overwrite files:
    #Ex. tile_images_from_dir('Paintings/Processed/Raw/Full/',M, output_size_cm, update = False)
    
    #Run for selected files: Does overwrite files
    #Ex. tile_images_from_dir('Paintings/Processed/Raw/Full/',M, output_size_cm,selected =selected_tifs)
    failed_to_tile = []
    
    samples = AbstractArtData(root_dir=read_dir)
    write_dir = Path(Path(read_dir).parents[0],write_folder)
    if update and os.path.isdir(write_dir):
        # print(read_dir,write_dir)
        samples.image_names = get_dir_core_differences(read_dir,write_dir)
    if selected: #just grab a few selected files to tile
        if isinstance(selected,str):
            samples.image_names = [selected]
        else:
            samples.image_names = selected
    
    for sample in samples:     
        file_name = sample['name'].split('_cropped')[0]
        if file_name in master['file'].unique():
            print(file_name)
            output_size_px = np.min((master.loc[master['file'] == file_name]['height_px'],master.loc[master['file'] == file_name]['width_px']))
            tile_an_image(sample,output_size_px,write_dir=write_dir,image_size=image_size)
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

def index_containing_substring(the_list, substring):
    index = []
    for i, s in enumerate(the_list):
        if isinstance(s,str):
            if substring in s:
                index.append(i)
    return index

def df_field_contains_substring(df,field,substring):
    return df.iloc[index_containing_substring(df[field], substring)]