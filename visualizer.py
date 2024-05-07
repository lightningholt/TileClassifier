from fastai.vision.all import *
from fastai.data.all import *
import pandas as pd
import cv2
import util
from util import load_metrics,get_paintings_list,get_max_output_size_px_based,str_round
from slice_classes import AbstractArtData, SliceCrop
from PIL import Image,ImageOps
import make_report_data
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from skimage.transform import resize as Resize
from sklearn.metrics import mean_squared_error
from plotting import make_row_hist_combined

def create_onepainting(results,file_start,slice_size):
    # print(results.file.str.startswith(file_start))
    # assert isinstance(slice_size,str), 'slice size needs to be a string!'
    slice_size = str(slice_size)
    one_painting = results[results.file.str.startswith(file_start +'_')]
    # print(one_painting)
    one_painting=one_painting[one_painting.slice_size==slice_size]
    # print('one_painting',one_painting)
    column=one_painting.apply(lambda row: int(row.file.split('_C')[1].split('_')[0]),axis=1)
    row=one_painting.apply(lambda row: int(row.file.split('_R')[1].split('.')[0]),axis=1)
    # print(len(one_painting.index))
    assert len(one_painting.index) > 0, 'no rows for painting ' + file_start + ' at slice size ' + slice_size +' found'
    one_painting['C']=column
    one_painting['R']=row
    column2=one_painting.apply(lambda row: ('C'+row.file.split('_C')[1].split('_')[0]),axis=1)
    row2=one_painting.apply(lambda row: ('R'+row.file.split('_R')[1].split('.')[0]),axis=1)
    one_painting['C2']=column2
    one_painting['R2']=row2
    
    one_painting.sort_values('file',inplace=True)
    
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

def colorize(files,image,one_painting,colors,transparency = [[0.5,0.5],[0.5,0.5]]):
    
    
    # print('files',files)
    crow=one_painting[one_painting['file'] == files]
    # print('crow',crow)
    # print(one_painting)
    # print(crow.pollock_prob.iloc[0])
    
    if isinstance(colors,tuple): #dynamically set color
        # file = apply_color(image,colors,transparency= [1,np.abs(1*crow.actual.iloc[0]-crow.pollock_prob.iloc[0])])
        file = apply_color(image,colors,transparency= [1,1-crow.pollock_prob.iloc[0]])
    else:
        if crow['prediction'].values[0]==crow['actual'].values[0]:
            file = apply_color(image,colors[0],transparency= transparency[0])
        else:
            file = apply_color(image,colors[1],transparency= transparency[1])
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
def image_visualizer(results,slice_size,filename,savename,image_base_path = 'Paintings/Processed/Descreened',colors = (0,0,255),savepath = '.',transparency = [[1,0.2],[1,0.2]],resize = False):
    # colors = [(0,255,0),(0,0,255)] [(green),(red)] gives two static colors with respective transparency
    # colors = (0,0,255) (red) gives one dynamically set color based on pollock prob. Input transparency is ignored
    dir_name = Path(image_base_path,slice_size)
    results.slice_size = results.slice_size.astype(str) #converts to str for consitancy
    one_painting=create_onepainting(results,filename,slice_size)
    # print(one_painting)
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
        
        # print('counter',columns[counter])
        files = sorted([file for file in os.listdir(dir_name) if file.startswith(filename + '_') and ('_'+columns[counter]+'_') in file])
        # print(files)
        
        images=[]
        
        length = len(files)
        for i in range(length):
            file = os.path.join(dir_name,files[i])
            tempfile=files[i]
            # print(tempfile)
            img = cv2.imread(file)
            img=colorize(tempfile,img,one_painting,colors,transparency = transparency)
            # print(file)
            images.append(img)
            # print(images[-1].shape)
        himages=concat_images(images, 'horizontal', with_plot=False)
        # print('himages',himages.shape)
        vimages.append(himages)
        
        counter+=1
    final_image=concat_images(vimages, 'vertical', with_plot=False)
    # print('fimages',final_image.shape)
    if savepath=='none':
        print('not saved')
    else:
        if resize:
            w = resize
            w_percent = w/final_image.shape[1]
            h_size = int((float(final_image.shape[0]) * float(w_percent)))
            
            final_image = cv2.resize(final_image, dsize=(w,h_size), interpolation=cv2.INTER_CUBIC)
        # print(type(final_image))
        # PIL_im = Image.fromarray(final_image)
        # PIL_im.save(os. path. join(savepath , savename))
        cv2.imwrite(os. path. join(savepath , savename), final_image)
    return final_image

def image_visualizer_wrapper(run_folder,paintings = 'incorrect',container = 'runs',save_ending ='.png',save_folder = 'viz',image_base_path = 'Paintings/Processed/Raw',colors = (0,0,255),transparency = [[1,0.2],[1,0.2]],resize=False):
    # colors = [(0,255,0),(0,0,255)] [(green),(red)] gives two static colors with respective transparency
    # colors = (0,0,255) (red) gives one dynamically set color based on pollock prob. Input transparency is ignored
    results = pd.read_csv(Path(container,run_folder, 'results.csv'),low_memory=False)
    metrics = load_metrics(Path(container,run_folder, 'metrics.csv'))
    filenames = get_paintings_list(metrics,paintings=paintings)

    for filename in filenames:
        print(filename)
        # print(results[results.file.str.startswith(filename +'_')])
        sizes = sorted(list(set(results[results.file.str.startswith(filename +'_')].slice_size.tolist())))
        print(len(sizes))
        print(sizes)
        for slice_size in sizes:
            savepath = Path(container,run_folder,save_folder,filename)
            savename = slice_size+save_ending
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            image_visualizer(results,slice_size,filename,savename,image_base_path = image_base_path,colors = colors,savepath = savepath,transparency = transparency,resize=False)
            
            
def make_as_viable_grayscale(matrix):
    return Image.fromarray((matrix*255.9999).astype(np.uint8))

def get_single_image_array(file_name,read_dir):
    samples = AbstractArtData(root_dir=read_dir)
    selected = file_name + '_cropped.tif'
    samples.image_names = [selected]
    for sample in samples:
        image = sample['image']   
    return image

def get_output_size_px(master,file_name,output_size_cm):
    if output_size_cm == 'Max':
        output_size_px = np.min([master.loc[master['file'] == file_name]['width_px'].iloc[0],master.loc[master['file'] == file_name]['height_px'].iloc[0]])
    else:
        resolution = float(master.loc[master['file'] == file_name]['px_per_cm_height'])
        output_size_px = round(output_size_cm * resolution)
    output_size_px = SliceCrop(int(output_size_px),0).output_size #throw in to SliceCrop class to get output_size formatted
    # print(output_size_px)
    return output_size_px

def get_im_shape_from_master(master,file_name):
    return (int(master[master.file==file_name].height_px),int(master[master.file==file_name].width_px))

def get_pollock_prob_matrix(file_name,output_size_cm,output_size_px ,results,master,gray = False,nan=True,verbose = False):
    # image = get_single_image_array(file_name,read_dir)  
    # output_size_px = get_output_size_px(master,file_name,output_size_cm)
    im_shape = get_im_shape_from_master(master,file_name)
    start_loc = util.get_tile_start(im_shape,output_size_px)
    results.slice_size = results.slice_size.astype(str) #converts to str for consitancy
    # print(output_size_cm)
    one_painting=create_onepainting(results,file_name,output_size_cm).sort_values('file')
    assert len(one_painting) > 0, 'no matrix found'
    #reshape pollock probs into matrix
    cs = np.max(one_painting.C)+1
    rs = np.max(one_painting.R)+1
    prob_matrix_reduced = np.array(one_painting.pollock_prob).reshape(cs,rs)
    #get pollock prob matrix 
    tile_matrix_base = np.ones(output_size_px)
    prob_matrix = np.kron(prob_matrix_reduced,tile_matrix_base)
    #pad pollock prob matrix to size of orginal image
    top_pad = start_loc[0]
    bottom_pad = im_shape[0] - prob_matrix.shape[0] - top_pad
    left_pad = start_loc[1]
    right_pad = im_shape[1] - prob_matrix.shape[1]-left_pad
    if verbose:
        print('Top_pad',top_pad)
        print('bottom_pad',bottom_pad)
        print('left_pad',left_pad)
        print('right_pad',right_pad)
    
    
    if nan == True:
        padded = np.pad(prob_matrix, pad_width=((top_pad, bottom_pad), (left_pad, right_pad)),constant_values=np.nan)
        ones = np.pad(np.ones(prob_matrix.shape),pad_width=((top_pad, bottom_pad), (left_pad, right_pad)),constant_values=np.nan)
    else:
        padded = np.pad(prob_matrix, pad_width=((top_pad, bottom_pad), (left_pad, right_pad)),constant_values=0)
        ones = np.pad(np.ones(prob_matrix.shape),pad_width=((top_pad, bottom_pad), (left_pad, right_pad)),constant_values=0)
    if gray:
        padded = make_as_viable_grayscale(padded)
        ones = make_as_viable_grayscale(ones)
    # print(padded)
    return padded,ones
def get_pollock_prob_matrix_overlay(file_name,output_size_cm,output_size_px,results,master,gray = False,add = 0,verbose = True):

    # image = get_single_image_array(file_name,read_dir)  
    # output_size_px = get_output_size_px(master,file_name,output_size_cm)
    results.slice_size = results.slice_size.astype(str) #converts to str for consitancy
    im_shape = get_im_shape_from_master(master,file_name)
    util.print_if(verbose,'im_shape=',im_shape)
    util.print_if(verbose,'output_size_px=',output_size_px)
    one_painting=create_onepainting(results,file_name,output_size_cm).sort_values('file')
    assert len(one_painting) > 0, 'no matrix found'
    #reshape pollock probs into matrix
    cs = np.max(one_painting.C)+1
    rs = np.max(one_painting.R)+1
    prob_matrix_reduced = np.array(one_painting.pollock_prob).reshape(cs,rs)
    #get pollock prob matrix 
    util.print_if(verbose,'prob_matrix_reduced.shape=',prob_matrix_reduced.shape)


    # prob_matrix = np.kron(prob_matrix_reduced,tile_matrix_base)
    # num_slice,slice_locs_0,slice_locs_1 = util.get_num_tiles(im_shape,output_size_px,add = add)
    num_slice,slice_locs_0,slice_locs_1 = util.get_num_tiles(im_shape,output_size_px,num_slices = [cs,rs],add = add)
    util.print_if(verbose,'num_slice=',num_slice)
    tile_overlays = []

    for i in range(num_slice[0]):
        for j in range(num_slice[1]):
            # if np.max(output_size_px) <= np.min(im_shape):
            tile_matrix = np.ones(output_size_px)*prob_matrix_reduced[i,j]
            # else:
            #     tile_matrix = np.ones((np.min(output_size_px),np.min(output_size_px)))*prob_matrix_reduced[i,j]
            tile_overlays.append(util.set_tile_on_nan_matrix(im_shape[0:2],tile_matrix,(slice_locs_0[i],slice_locs_1[j])))
    layer = np.nanmean(tile_overlays,axis=0)
    ones = np.ones(layer.shape)
    if gray:
        layer = make_as_viable_grayscale(layer)
        ones = make_as_viable_grayscale(ones)
    return layer,ones

def resize_master_painting(master,file_name,resize):
    if isinstance(resize,tuple):
        im_size_orig = get_im_shape_from_master(master,file_name)
        resize = get_resize_scale_factor_within_bounds(im_size_orig, max_size=resize)
        # print('resize',resize)
    master.width_px = (master.width_px*resize).round()
    master.height_px = (master.height_px*resize).round()
    master.px_per_cm_height = master.height_px/master.height_cm
    master.px_per_cm_width = master.width_px/master.width_cm   
    return master 

def get_combined_pollock_prob_matrix(file_name,results,master_file,gray = False,nan=True,center_tile = True,resize = False,add = 0,path = 'viz/P86(S)',save_name = False,verbose = True):  
    master = master_file.copy()
    # max_output_size_cm = get_max_output_size_px_based(master,file_name)# int(min(master[master.file == file_name].height_cm.iloc[0],master[master.file == file_name].width_cm.iloc[0]))
    max_output_size_cm = util.get_max_size_from_results(results,file_name)
    # check to see if max output size in cm doesn't line up with pixels. Set max to fit the pixel value
    # print(max_output_size_cm)
    sizes = range(10, max_output_size_cm,5)
    if len(sizes) == 0: #catches the instance where the max_output_size_cm is 10
        sizes = [10]
    assert get_max_output_size_px_based(master,file_name) >= 10, 'max output size is too small!' + str(get_max_output_size_px_based(master,file_name))
    
    if resize:
        master = resize_master_painting(master,file_name,resize)

    
    im_size = get_im_shape_from_master(master,file_name)
    # print(im_size)
    padded_size = ((len(sizes)+1,)+im_size)
    padded = np.empty(padded_size)
    padded[:] = np.nan
    one_matrix = np.empty(im_size)
    one_matrix[:] = np.nan
    for i,output_size_cm in enumerate(sizes):
        util.print_if(verbose,output_size_cm)
        output_size_px = get_output_size_px(master,file_name,output_size_cm)
        # print(output_size_px)
        # print(i)
        if center_tile:
            padded[i],ones = get_pollock_prob_matrix(file_name,output_size_cm,output_size_px,results,master,gray = False,nan=nan,verbose = verbose)
        else:
            padded[i],ones = get_pollock_prob_matrix_overlay(file_name,output_size_cm,output_size_px,results,master,gray = False,add = add,verbose = verbose)
        one_matrix = np.nansum([one_matrix,ones],axis=0)
    # print(max_output_size_cm)
    output_size_px = get_output_size_px(master,file_name,'Max')
    if center_tile:
        padded[i+1],ones = get_pollock_prob_matrix(file_name,'Max',output_size_px,results,master,gray = False,nan=nan,verbose = verbose)
    else:
        padded[i+1],ones = get_pollock_prob_matrix_overlay(file_name,'Max',output_size_px,results,master,gray = False,add = add,verbose = verbose)
    one_matrix = np.nansum([one_matrix,ones],axis=0)
    # print('before combined')
    # combined_array=np.stack(padded,axis=0) #Doesn't seem to be necessary
    util.print_if(verbose,'number of layers', len(sizes)+1)
    # print('before nanmean')
    combined = np.nanmean(padded,axis=0)
    # combined = np.nanmean(combined_array,axis=0)
    # combined = np.mean(combined_array,axis=0)
    # print('before nansum')
    # num_layer_matrix = np.nansum(one_matrix,axis=0)
    if gray:
        combined = make_as_viable_grayscale(combined)
        num_layer_matrix = make_as_viable_grayscale(num_layer_matrix/np.max(num_layer_matrix))
        layers = []
        for pad in padded:
            layers.append(make_as_viable_grayscale(pad))
        padded = layers
    if save_name:
        np.save(os.path.join(path,save_name),combined)
        util.print_if(True, 'saved ',os.path.join(path,save_name))
    return combined,padded,one_matrix


def process_combined_pollock_prob_matrix(combined_result,file_name,read_dir,save_path = False,contain = (2048,2048),line_mask = False,color = [0,0,0],extension = '.png',inverse_pollock_signatures = True,resize = False):
    selected = file_name + '_cropped.tif'
    combined = combined_result.copy()
    if inverse_pollock_signatures:
        combined[np.isnan(combined)]=1
        combined = 1-combined      
        cr = cv2.merge((combined*0,combined*0,combined*255))
    else:
        combined[np.isnan(combined)]=0
        cr = cv2.merge((combined*0,combined*255,combined*0))
        
    combined_gray = make_as_viable_grayscale(combined)
    cr = cr.astype(np.uint8)
    transparency = [1,0.5]
    alpha =  transparency[0]
    beta= transparency[1]
    # gray = cv2.imread(str(Path(read_dir,selected)))
    gray = cv2.imread(str(Path(read_dir,selected)),cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if resize:
        gray = cv2.resize(gray,cr.shape[:2][::-1])
    result = cv2.addWeighted(gray, alpha, cr,beta,0)

    

    
    if save_path:
        if contain:
            img_gray = resize_within_bounds(gray, max_size=contain)
            img_comb = resize_within_bounds(result, max_size=contain)
            img_cr = resize_within_bounds(cr, max_size=contain)
            if isinstance(line_mask,np.ndarray):
                line_mask = resize_within_bounds(line_mask, max_size=contain)
        else:
            img_gray = gray
            img_comb = result
            img_cr = cr
        if isinstance(line_mask,np.ndarray):
            img_gray = overlay_line_on_image(img_gray, line_mask,color = color)
            img_comb = overlay_line_on_image(img_comb, line_mask,color = color)
            img_cr = overlay_line_on_image(img_cr, line_mask,color = color)
        cv2.imwrite(os. path. join(save_path , 'gray_viz' + extension), img_gray)
        cv2.imwrite(os. path. join(save_path , 'combined_viz' + extension), img_comb)  
        
        #give the mask some transparency
        img_cr = cv2.cvtColor(img_cr, cv2.COLOR_BGR2BGRA)
        img_cr[:, :, 3] = int(beta * 255)
        
        # print('img_cr size', img_cr.shape)
        cv2.imwrite(os. path. join(save_path , 'filter_viz' + extension), img_cr)
        
        print(file_name+'_xxx_viz' + extension +' saved to '+save_path)
    return result,combined_gray
    # plt.imshow(result)
    
def process_layer_matrix(layer_matrix,file_name = '',save_path = False,contain = (2048,2048),extension = '.png'):
    layer_matrix = make_as_viable_grayscale(layer_matrix/np.max(layer_matrix))
    if save_path:
        if contain:
            img = ImageOps.contain(layer_matrix, contain)
        else:
            img = layer_matrix
        img.save(os. path. join(save_path , 'layer_matrix' + extension))
        print(file_name+'layer_matrix'+extension +' saved to '+save_path)    
    return layer_matrix

def resize_within_bounds(image, max_size=(2048, 2048)):
    """
    Resizes an image to fit within a specified bounding box while maintaining its aspect ratio.
    """
    height, width = image.shape[:2]
    max_height,max_width = max_size
    if height > max_height or width > max_width:
        scale_factor_w = max_width / width
        scale_factor_h = max_height / height
        scale_factor = min(scale_factor_w, scale_factor_h)
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        return resized_image
    else:
        return image
def get_resize_scale_factor_within_bounds(im_shape, max_size=(2048, 2048)):
    """
    Resizes an image to fit within a specified bounding box while maintaining its aspect ratio.
    """
    height, width = im_shape[:2]
    max_height,max_width = max_size
    if height > max_height or width > max_width:
        scale_factor_w = max_width / width
        scale_factor_h = max_height / height
        scale_factor = min(scale_factor_w, scale_factor_h)
        return scale_factor
    else:
        return 1
    
def generate_line_mask(image, thickness_ratio=0.01):
    # Get the height and width of the input image
    height, width = image.shape[:2]

    # Calculate the thickness of the black line based on the image size
    thickness = int(max(height, width) * thickness_ratio)

    # Convert the input image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to uint8
    image = cv2.convertScaleAbs(image)

    # Threshold the input image to create a binary image
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

    # Apply Canny edge detection to the binary image
    edges = cv2.Canny(binary_image, 0, 1)

    # Dilate the edges to create a thicker line
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Convert the dilated edges back to BGR color space
    line_mask = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)

    return line_mask


def overlay_line_on_image(image, line_mask, color = [0,0,0]):
    # Convert the input image to RGB if it's grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize the black line image to match the size of the input image
    line_mask = cv2.resize(line_mask, (image.shape[1], image.shape[0]))

    # Create a mask from the black line image by thresholding it
    _, mask = cv2.threshold(line_mask, 1, 255, cv2.THRESH_BINARY)

    # Set the pixels in the result image to color where the mask is non-zero
    image = np.where(mask != 0, color, image)
    result = cv2.convertScaleAbs(image)

    return result

def write_individual_slice_layers(padded,file_name,read_dir,
                                  save_path = False,
                                  contain = (2048,2048),
                                  line_mask = False,
                                  color = [0,0,0],
                                  thickness_ratio = 0.01,
                                  extension = '.png',
                                  inverse_pollock_signatures = True,
                                  resize = False,
                                  cmap = False):

    
    save_path = os.path.join(save_path , 'slice_layers')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    selected = file_name + '_cropped.tif'
    gray = cv2.imread(str(Path(read_dir,selected)),cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if resize:
        gray = cv2.resize(gray,padded[0].shape[:2][::-1])
    for i,layer in enumerate(padded.copy()):
        if line_mask:
            arr = layer.copy()
            mask = ~np.isnan(arr)
            arr[mask] = 0
            arr[np.isnan(layer)]=1
            mask = generate_line_mask(arr,thickness_ratio = thickness_ratio)
        
        # print(layer)
        if cmap:
            result = overlay_cmap_on_image(layer,cmap,file_name, read_dir_full_cropped = read_dir,write_gray=False,save=False,show_figure = False)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        else:
            if inverse_pollock_signatures:
                layer[np.isnan(layer)]=1
                layer = 1-layer
                cr = cv2.merge((layer*0,layer*0,layer*255))
            else:
                layer[np.isnan(layer)]=0
                cr = cv2.merge((layer*0,layer*255,layer*0))
            padded_gray = make_as_viable_grayscale(layer)
            cr = cr.astype(np.uint8)
            transparency = [1,0.5]
            alpha =  transparency[0]
            beta= transparency[1]
            result = cv2.addWeighted(gray, alpha, cr,beta,0)

        if contain:
            # img_gray = resize_within_bounds(gray, max_size=contain)
            img_comb = resize_within_bounds(result, max_size=contain)
            # img_cr = resize_within_bounds(cr, max_size=contain)
            if line_mask:
                mask = resize_within_bounds(mask, max_size=contain)
        else:
            # img_gray = gray
            img_comb = result
            # img_cr = cr
        if line_mask:
            # img_gray = overlay_line_on_image(img_gray, line_mask,color = color)
            img_comb = overlay_line_on_image(img_comb, mask,color = color)
            # img_cr = overlay_line_on_image(img_cr, line_mask,color = color)
        # cv2.imwrite(os. path. join(save_path , file_name+'_gray_viz.tif'), img_gray)

        if i == len(padded)-1:
            write_path = os.path.join(save_path,'combined_viz_' + 'Max'+extension)
            pm_write_path = os.path.join(save_path,'pm_' + 'Max'+extension)
        else:
            write_path = os.path.join(save_path,'combined_viz_' + str(10+i*5)+extension)
            pm_write_path = os.path.join(save_path,'pm_' + str(10+i*5)+extension)
        cv2.imwrite(write_path, img_comb) 
        pollock_map(layer,hist_width = 0.4,blur=True,custom_cmap=True,save_path=pm_write_path,gradient = False,relative_radial = True,show_figure = False,save_individual=True)
        #give the mask some transparency
#         img_cr = cv2.cvtColor(img_cr, cv2.COLOR_BGR2BGRA)
#         img_cr[:, :, 3] = int(beta * 255)
        
#         cv2.imwrite(os. path. join(save_path , file_name+'_filter_viz.tif'), img_cr)            
    return

def generate_report_viz(read_dir = 'Paintings/Processed/Raw/Full/',
                        results = pd.read_csv('runs/gcp_classifier-3_10-Max_color_10-15-2022/results.csv'),
                        master =pd.read_parquet('master.parquet'),
                        file_name = 'C2',
                        path = False,
                        color = [0,255,255],
                        thickness_ratio = 0.001,
                        inverse_pollock_signatures = True,
                        center_tile = True,
                        resize = False,
                        write_individual_layers = True,
                        write_pollock_map = True,
                        add = 0,
                        pollock_map_colors = [(1, 0, 0), (0, 0, 0), (0, 0.5, 0)],
                        relative_radial = False,
                        show_figure = True,
                        verbose = True,
                        specify_viz_save = False,
                        write_individual_pollock_maps = True,
                        write_tile_hist = True
                        ):
    #set default path
    if not path:                   
        path = os.path.join('viz',file_name)
    if specify_viz_save:
        path = specify_viz_save
        data_path = path
    else:
        data_path = path    
    if not os.path.exists(path):
        os.makedirs(path)
    combined_result,padded,layer_matrix = get_combined_pollock_prob_matrix(file_name,results,master,gray = False,nan = True,center_tile = center_tile,resize = resize,add = add,path = data_path,save_name = 'combined.npy',verbose = verbose)
    # line_mask= generate_line_mask(layer_matrix,thickness_ratio = thickness_ratio)
    ##print(combined_result.shape,padded.shape,layer_matrix.shape)
    ###probably don't need to return the results of these next two
    # result,combined_gray = process_combined_pollock_prob_matrix(combined_result,file_name,read_dir,save_path = path,line_mask=line_mask,color = color,inverse_pollock_signatures=inverse_pollock_signatures,resize = resize )
    # LM = process_layer_matrix(layer_matrix,file_name = file_name, save_path = path)

    if inverse_pollock_signatures:
        cb_color = [0,0,255]
    else:
        cb_color = [0,255,0]
    # make_colorbar(path = path,save_name = 'color_bar.png',color = cb_color,size = (100,2048),beta = 0.5,Range = [0,1])
    cmap = get_custom_colormap(colors = pollock_map_colors)
    if write_individual_layers:
        write_individual_slice_layers(padded,file_name,read_dir,save_path = path,line_mask=True,color = color,thickness_ratio = thickness_ratio,inverse_pollock_signatures=inverse_pollock_signatures,resize = resize,cmap = cmap)
    if write_pollock_map:
        pollock_map(combined_result,hist_width = 0.4,blur = True,save_individual=True,custom_cmap=True,save_path=Path(path,'pollock_map.png'),gradient = False,colors =  pollock_map_colors,relative_radial = relative_radial,show_figure = show_figure)
        overlay_cmap_on_image(combined_result,cmap,file_name, read_dir_full_cropped = read_dir,save=Path(path,'overlay.png'),show_figure = show_figure)
        # make_euc_edge_hists(master,file_name,combined_result,bin_width_cm = 10,start_loc = False,path=path)
    if write_tile_hist:
        make_row_hist_combined(file_name,str(Path(read_dir).parent),results,
                                        save_pre_path=path,
                                        hist_save_name='row_hist.png',save_folder='',one_vote = True)

    return combined_result,padded

def calculate_mean_from_edge(array, start=2, end=5):
    """
    Calculate the mean of non-NaN pixels that are a certain distance away from the edge of a 2D array.

    Args:
        array (numpy.ndarray): 2D array.
        start (int): Starting distance from the edge (inclusive). Default is 2.
        end (int): Ending distance from the edge (inclusive). Default is 5.

    Returns:
        float: Mean of non-NaN pixels a certain distance away from the edge.
    """

    # Initialize an empty mask to store the valid pixels
    mask = np.zeros_like(array, dtype=bool)

    # Loop over the distance range (start to end) and update the mask for each edge
    if start == 0:
        mask[:, :] = True  # Top edge
    else:
        mask[start:-start, start:-start] = True  # Top edge
    mask[end:-end, end:-end] = False

    # Use the mask to extract the non-NaN pixels from the input array
    pixels = array[mask & ~np.isnan(array)]

    # Calculate the mean of the non-NaN pixels
    mean = np.nanmean(pixels)

    return mask, mean

def calculate_non_nan_fraction(data, bin_width,axis = 1,start = 0):
    """
    Calculate the fraction of non-NaN values in a set of columns of a 2D array using np.sum().

    Args:
        data (numpy.ndarray): 2D array containing NaNs and values.
        bin_width (int): Number of columns to consider together as a single bin.

    Returns:
        numpy.ndarray: 1D array containing the fraction of non-NaN values in each bin.
    """
    # Get the number of columns in the data
    num_cols = data.shape[axis]

    # Create an array to store the fraction of non-NaN values in each bin
    fractions = np.empty(num_cols // bin_width)


    # Loop over the bins and calculate the fraction of non-NaN values in each bin
    for i in range(num_cols // bin_width):
        # start_col = i * bin_width
        # end_col = (i + 1) * bin_width
        if axis == 0: 
            cols = data[start:(start+bin_width),:]
        elif axis == 1:
            cols = data[:, start:(start+bin_width)]
        else:
            print('invalid axis value')
        fractions[i] = np.nanmean(cols)
        start = start + bin_width

    return fractions

#not using this as of 6-19?
def plot_fractions(fractions,xticks,xlabel = 'Distance (cm)',path = False,save_name = 'plot',axis = 0,im_size = False):
    # Create bar plot
    # Set the color of the bars
    bar_color = 'blue'  # Adjust this value to change the color of the bars

    # Set the color of the outline (edgecolor) to black
    edge_color = 'black'
    bar_width = xticks[1]-xticks[0]
    if axis == 0:
        # 
        plt.barh(xticks, fractions, height=bar_width, align='edge', color=bar_color, edgecolor=edge_color) # Use barh for horizontal bar plot
        plt.xlim(0, 1)
        plt.yticks([])
        plt.xlabel('Average Pollock Signature')
        if not im_size:
            plt.ylim(min(xticks),max(xticks))
        else:
            plt.ylim(0,im_size[axis])
        # plt.ylabel('Distance(cm)')
    else:
        plt.bar(xticks, fractions, width=bar_width,align='edge', color=bar_color, edgecolor=edge_color)
        # Set x-axis labels
        plt.xticks([])
        plt.ylim(0, 1)
        if not im_size:
            plt.xlim(min(xticks),max(xticks))
        else:
            plt.xlim(0,im_size[axis])
        plt.ylabel('Average Pollock Signature')
        # plt.xlabel(xlabel)
    


    if path:
        dpi = 80  # Specify the DPI (dots per inch)
        # width, height = im_size[0], im_size[1]  # Specify the width and height of the figure in pixels
        plt.savefig(os.path.join(path,save_name + '.png'), dpi=dpi,bbox_inches='tight', pad_inches=0)

        # plt.savefig(os.path.join(path,save_name + '.png'),dpi = image_height)
        print('Plot saved to ' + os.path.join(path,save_name + '.png'))
        
    # Show the plot
    plt.show()

#not using this as of 6-19
def plot_edge_fractions(fractions,xticks,xlabel = 'Distance From Edge(cm)',path = False,save_name = 'plot'):
    # Create bar plot
    # Set the color of the bars
    bar_color = 'blue'  # Adjust this value to change the color of the bars

    # Set the color of the outline (edgecolor) to black
    edge_color = 'black'
    
    bar_width = xticks[1]-xticks[0]
    plt.bar(xticks, fractions, width=bar_width,align='edge', color=bar_color, edgecolor=edge_color)
    # Set x-axis labels
    plt.xticks(xticks[::2])
    plt.ylim(0, 1)
    plt.ylabel('Average Pollock Signature')
    plt.xlabel(xlabel)
    


    if path:
        plt.savefig(os.path.join(path,save_name + '.png'))
        print('Plot saved to ' + os.path.join(path,save_name + '.png'))
        
    # Show the plot
    plt.show()
    

def from_edge_distributions(master,file_name,combined_result,bin_width_cm = 5,path=False,save_name = 'edge'):
    bin_width = get_output_size_px(master,file_name,bin_width_cm)[0]
    if combined_result.shape != get_im_shape_from_master(master,file_name): #if the image  was resized
        bin_width = int(bin_width * combined_result.shape[0]/get_im_shape_from_master(master,file_name)[0])
    bins = range(0,min(combined_result.shape),bin_width)
    bin_len = int(np.floor(len(bins)/2))
    bins = bins[0:bin_len]
    mean = np.empty(len(bins))
    masks = []
    cm_sizes = [i*bin_width_cm for i in range(len(bins))]
    for i,start in enumerate(bins):
        mask,mean[i] = calculate_mean_from_edge(combined_result, start=start, end=start + bin_width)
        masks.append(mask)
    plot_edge_fractions(mean,cm_sizes,path=path,save_name = save_name)
    return mean,masks


#not using this as of 6-19
def euclidean_distribution(master,file_name,combined_result,bin_width_cm = 10,axis = 1,start_loc = 'centered',path=False,save_name = 'euclidean'):
    bin_width = get_output_size_px(master,file_name,bin_width_cm)
    if combined_result.shape != get_im_shape_from_master(master,file_name): #if the image  was resized
        bin_width = tuple(int(element * combined_result.shape[0]/get_im_shape_from_master(master,file_name)[0]) for element in bin_width)

    if start_loc == 'centered':
        start_loc = util.get_tile_start(combined_result,bin_width)
    else:
        start_loc = (0,0)
    fractions = calculate_non_nan_fraction(combined_result,bin_width[0],axis = axis,start = start_loc[axis])
    print(bin_width)
    print(fractions)
    bins = range(start_loc[axis],combined_result.shape[axis],bin_width[0])
    print(bins)
    # cm_sizes = [i*bin_width_cm for i in range(len(bins))]
    print(len(bins))
    print(fractions.shape)
    plot_fractions(fractions,bins[:-1],path=path,save_name = save_name +'_' + str(axis),axis=axis,im_size = combined_result.shape)
    return fractions
# def select_evenly_spaced_numbers(arr, num_numbers):
#     """
#     Select a specified number of evenly spaced numbers from an array.

#     Args:
#         arr (array-like): Input array from which numbers are selected.
#         num_numbers (int): Number of evenly spaced numbers to select.

#     Returns:
#         list: List of selected numbers.
#     """
#     # Calculate step size
#     step_size = len(arr) // num_numbers

#     # Initialize variables
#     selected_numbers = []
#     current_index = 0

#     # Loop for the specified number of numbers
#     for i in range(num_numbers):
#         # Select element at current index
#         selected_numbers.append(arr[current_index])

#         # Increment current index by step size
#         current_index += step_size

#     return selected_numbers

#not using this as of 6-19
def make_colorbar(path = 'viz/P86(S)',save_name = False,color = [0,0,255],size = (100,2048),beta = 0.5,Range = [0,1]):
    x = np.linspace(Range[0], Range[1], size[1])
    array = np.tile(x, (size[0], 1))
    combined = array #np.ones_like(combined_result)
    cmax = cv2.merge((combined*color[0],combined*color[1],combined*color[2]))
    cmax = cmax.astype(np.uint8)
    # img_cr = vz.resize_within_bounds(cmax, max_size=(2048,2048))
    img_cr = cv2.cvtColor(cmax, cv2.COLOR_BGR2BGRA)
    img_cr[:, :, 3] = int(beta * 255)
    if save_name:
        cv2.imwrite(os. path. join(path, save_name), img_cr)
    return img_cr

#not using this as of 6-19
def make_euc_edge_hists(master,file_name,combined_r,bin_width_cm = 10,start_loc = False,path='viz/example'):
    # bin_width = 10
    E = []
    for i in range(2):
        E.append(euclidean_distribution(master,file_name,combined_r,bin_width_cm = bin_width_cm,axis = i,start_loc = False,path=path,save_name = 'euclidean_hist_'+str(bin_width_cm)))
    R,masks = from_edge_distributions(master,file_name,combined_r,bin_width_cm = bin_width_cm,path=path,save_name = 'from_edge_hist_'+str(bin_width_cm))  
    return masks

def process_image(testing_master,
                file_name,
                center_tile = False,
                tile_update = False,
                start = 10,
                read_dir_full_cropped = 'Paintings/Processed/Raw/Full',
                learner_folder = 'gcp_classifier-3_10-Max_color_10-15-2022',
                select_painting = 'P2(V)', #only for comparison if making report
                inverse_pollock_signatures = True,
                resize = (1000,1000),
                write_individual_layers=False,
                viz_path_start = 'viz',
                viz_path_end = 'overlap_resize',
                save_pre_path = 'painting_preds',
                save_folder_description = 'overlap',
                make_report = False,
                do_tiling = True,
                do_report_data = True,
                do_viz = True,
                add = 0,
                pollock_map_colors = [(1, 0, 0), (0, 0, 0), (0, 0.5, 0)],
                write_pollock_map = True,
                relative_radial = False,
                show_figure = False,
                verbose = False,
                specify_viz_save = False,
                item_tfms = False
                ):
    testing_master = testing_master[testing_master.file == file_name]
    viz_path = os.path.join(viz_path_start,file_name+'_' +viz_path_end)
    learner_master = pd.read_parquet('master.parquet')  #only would need to change this if we trained on different paintings
    if do_tiling:
        print('running tiler for',file_name)
        util.tile_images(testing_master,read_dir_full_cropped,center_tile = center_tile,update = tile_update,start = start,add=add,verbose = verbose)
    if do_report_data:
        print('making report (data) for',file_name)
        if not center_tile:
            if add > 0:
                img_path_testing = str(Path(read_dir_full_cropped).parent)+'-overlap' + '-' +str(add)
            else:
                img_path_testing = str(Path(read_dir_full_cropped).parent)+'-overlap'
        else:
            img_path_testing = str(Path(read_dir_full_cropped).parent)
        make_report_data.make_report_data(testing_master, learner_master,
                                    container = 'runs',
                                    learner_folder = learner_folder,
                                    img_path =  read_dir_full_cropped,
                                    img_path_testing =  img_path_testing,
                                    save_pre_path = save_pre_path ,
                                    save_folder = file_name+ '_' + save_folder_description+'_' + learner_folder,
                                    do_preds = True,
                                    one_vote = True,
                                    select_painting = select_painting,
                                    round_to = 2,
                                    make_report=make_report,
                                    verbose = verbose,
                                    item_tfms = item_tfms
                                    )
    if do_viz:
        results = pd.read_csv(Path(save_pre_path, file_name+'_'+ save_folder_description+'_'+learner_folder,'results.csv'))
        print('making viz for', file_name)
        combined_r,padded = generate_report_viz(read_dir = read_dir_full_cropped,
                                results = results,
                                master =testing_master,
                                file_name = file_name,
                                path = viz_path,
                                color = [0,255,255],
                                thickness_ratio = 0.005,
                                inverse_pollock_signatures = inverse_pollock_signatures,
                                center_tile=center_tile,
                                resize = resize,
                                write_individual_layers=write_individual_layers,
                                add = add,
                                pollock_map_colors=pollock_map_colors,
                                write_pollock_map=write_pollock_map,
                                relative_radial = relative_radial,
                                show_figure = show_figure,
                                verbose = verbose,
                                specify_viz_save = specify_viz_save
                                )
            
        

def pollock_map(combined_result,
                cbar_buffer =0.15,                
                # cbar_buffer = 0.2,
                hist_width = 0.4,
                custom_cmap = False,
                save_path = False,
                blur = True,
                gradient = False,
                colors = [(1, 0, 0), (0, 0, 0), (0, 0.5, 0)],
                relative_radial = False,
                bin_width_px = 1,
                show_figure = True,
                radial = False,
                threshold = 0.56,
                rescale = True,
                round_to = 2,
                save_individual = False,
                kernel_size = 51,
                sigma = 500
                ):
    plt.ioff()

    cr = combined_result.copy()
    if blur:
        # combined_result = gaussian_blur(combined_result,kernel_size = 51,sigma = 500)
        combined_result = gaussian_blur(combined_result,kernel_size = kernel_size,sigma = sigma)

    radial_matrix = get_radial_from_combined(combined_result,bin_width_px = bin_width_px)
    # Create a Figure, which doesn't have to be square.
    fig = plt.figure(facecolor='white')

    if custom_cmap:
        # cmap = get_custom_colormap(colors =  [(1, 0, 0,0.5), (0, 0, 0,0.5), (0, .5, 0,0.5)])
        cmap = get_custom_colormap(colors =  colors)
        cmap_radial = plt.cm.get_cmap('viridis')
    else:
        cmap = plt.cm.get_cmap('viridis') 
        cmap_radial = plt.cm.get_cmap('viridis')

    ax = fig.add_gridspec().subplots()
    ax.set(aspect=1)

    cr_aspect = cr.shape[0]/cr.shape[1]
    adjust_num = util.custom_squish_mapping(cr_aspect, 1, 0.5)
    cbar_buffer_coords = cbar_buffer / adjust_num


    #get the aspect ratio
    # a = combined_result.shape[0]/combined_result.shape[1] #only use if want a square UR area
    a=1

    ax_histx = ax.inset_axes([0, 1., 1, hist_width], sharex=ax)
    ax_histy = ax.inset_axes([1, 0, hist_width*a, 1], sharey=ax)
    # ax_cbar = ax.inset_axes([0, 1+hist_width+cbar_buffer, 1 + hist_width*a, .10])
    ax_cbar = ax.inset_axes([0, 1 + hist_width + cbar_buffer_coords, 1 + hist_width * a, 0.1])
    # ax_cbar2 = ax.inset_axes([0, 1+hist_width+cbar_buffer, 1 + hist_width*a, .10])
    ax_radial = ax.inset_axes([1,1,hist_width*a,hist_width])


    ax.tick_params(axis="both", labelbottom=False,labelleft=False, bottom=False, left=False)
    ax_histx.tick_params(axis="both", labelbottom=False,labelleft=True, bottom=False, left=True)
    ax_histy.tick_params(axis="both", labelbottom=True,labelleft=False, bottom=True, left=False)
    ax_radial.tick_params(axis="both", labelbottom=False,labelleft=False, bottom=False, left=False)
    ax_cbar.tick_params(axis="y", labelleft=False,left = False,labelright=False,right = False)
    ax_histy.set_xticks([0,threshold,1])
    ax_histy.set_xticklabels(['0',str(threshold),'1'])
    ax_histx.set_yticks([0,threshold,1])
    ax_histx.set_yticklabels(['0',str(threshold),'1'])


    y0 = np.nanmean(combined_result,axis = 0)
    x0=range(len(y0))
    points = np.array([x0, y0]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    if gradient:
        norm = plt.Normalize(0, 1)
    else:
        norm = BoundaryNorm([0, threshold, 1], cmap.N)
    
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(y0)
    lc.set_linewidth(2)
    line = ax_histx.add_collection(lc)
    # ax_histx.plot(y)
    
    
    y1 = np.nanmean(combined_result,axis = 1)
    x1= range(len(y1))
    points = np.array([y1, x1]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(y1)
    lc.set_linewidth(2)
    ax_histy.add_collection(lc)
    # ax_histy.plot(y,x)

    dummy_im = plt.imshow(np.array([[0,1]]),cmap = cmap)
    dummy_im.set_visible(False)

    min_val = np.min(combined_result)
    max_val = np.max(combined_result)
    if custom_cmap:
        combined_result = cmap(combined_result)
        if save_individual:
            path_individual = Path(Path(save_path).parent,'I' + Path(save_path).stem + '_' + str(kernel_size) + '_' + str(sigma) + '.png')
            plt.imsave(path_individual,combined_result)
            print('saved ' + str(path_individual))
        if not relative_radial:
            radial_matrix = cmap(radial_matrix)
        im = ax.imshow(combined_result,cmap = cmap)
        ax_histy.set_xlim(-0.1,1.1)
        ax_histx.set_ylim(-0.1,1.1)
    else:
        im = ax.imshow(combined_result)
        ax_histy.set_xlim(0.95*np.min(y1),1.05*np.max(y1))
        ax_histx.set_ylim(0.95*np.min(y0),1.05*np.max(y0))
    # ax_cbar.imshow(cb,aspect='auto')
    

    cb = plt.colorbar(dummy_im, cax=ax_cbar,orientation = 'horizontal')



    # Add twin axes 
    # ax_cbar2 = ax_cbar.twiny()

    # set upper ticks
    # ax_cbar2.set_xticks([min_val,max_val])
    # ax_cbar2.set_xticklabels(['min','max'])
    # set lower ticks
    cb.set_ticks([0,threshold,1])
    cb.set_ticklabels(['0',str(threshold),'1'])

    fs=12
    if radial:
        if relative_radial:
            ax_radial.imshow(radial_matrix,aspect = 'auto',cmap = cmap_radial)
        else:
            ax_radial.imshow(radial_matrix,aspect = 'auto')
    else:
        x_pos = 0.5  # 0.5 means the center of the x-axis
        y_pos = 0.5  # 0.5 means the center of the y-axis
        C = cr[cr > threshold].size/cr.size
        y = [np.nan,np.nan]
        for i in range(2):
            y[i] = np.nanmean(cr,axis = i)
            y[i] = np.sqrt(mean_squared_error(y[i],np.full(len(y[i]), np.mean(y[i]))))
            if rescale:
                y[i] = 1 - (2 * (y[i] - 0))
        U = np.mean(y)
        # Add the text to the center of the axes
        ax_radial.text(x_pos, y_pos - 0.15, 'U=' + str_round(U,round_to), ha='center', va='center', fontsize=fs, fontweight='bold')
        ax_radial.text(x_pos, y_pos + 0.15, 'C=' + str_round(C,round_to), ha='center', va='center', fontsize=fs, fontweight='bold')


    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300) 
        print('saved ' + str(save_path))
    if show_figure:
        plt.show()
    else:
        plt.close()
    if radial:
        return radial_matrix
    else:
        return C,U
    
def get_custom_colormap(colors = [(0, 0, 1), (0, 0, 0), (0, 0.5, 0)],positions = [0, 0.56, 1]):
    # Define the custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, colors)))
    return cmap

def from_edge_distributions_px(combined_result,bin_width_px = 1,path=False,save_name = 'edge'):
    bin_width = bin_width_px
    bins = range(0,min(combined_result.shape),bin_width)
    bin_len = int(np.ceil(len(bins)/2))
    bins = bins[0:bin_len]
    mean = np.empty(len(bins))
    masks = []
    for i,start in enumerate(bins):
        mask,mean[i] = calculate_mean_from_edge(combined_result, start=start, end=start + bin_width)
        masks.append(mask)
    
    # plot_edge_fractions(mean,cm_sizes,path=path,save_name = save_name)
    return mean,masks

def overlay_cmap_on_image(combined_result,
                          cmap, 
                          file_name,
                          read_dir_full_cropped = 'max_overlap/Processed/Raw/Full',
                          save = 'test2.png',
                          write_gray = True,
                          show_figure = True
                          ):
    plt.ioff()
    figure = plt.figure(facecolor='white')
    transparency = [1,0.5]
    alpha =  transparency[0]
    beta= transparency[1]
    blurred_image = gaussian_blur(combined_result)
    cr = (cmap(blurred_image)[:,:,:3]*128).astype('uint8')
    gray = cv2.imread(str(Path(read_dir_full_cropped,file_name + '_cropped.tif')),cv2.IMREAD_GRAYSCALE)
    
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    aspec_gray = round(gray.shape[0]/gray.shape[1],1)
    aspect_cr = round(cr.shape[0]/cr.shape[1],1)
    if not aspec_gray == aspect_cr:
        print("Warning: aspect ratio of the result and original image don't match. Confirm the file_name matches the combined_result: result = " + str(aspec_gray) + " and original = " + str(aspect_cr))
    gray = cv2.resize(gray,cr.shape[:2][::-1])
    result = cv2.addWeighted(gray,alpha, cr[:,:,:3],beta,0)
    im = plt.imshow(result,cmap = cmap)
    plt.axis('off')
    if save:
        save = Path(save)
        plt.savefig(save, bbox_inches='tight', dpi=300) 
        print('saved ' + str(save))
    if save and write_gray:
        cv2.imwrite(str(Path(save.parent,'gray.png')),gray)
        print('saved ' + str(Path(save.parent,'gray.png')))
    if show_figure:
        plt.show()
    else:
        plt.close()
    
    return result

def gaussian_blur(matrix,kernel_size = 51,sigma = 500):
    blurred_image = cv2.GaussianBlur(matrix, (kernel_size, kernel_size), sigma)
    return blurred_image

def get_radial_from_combined(combined_result,bin_width_px = 1):
    radial_matrix = np.full(combined_result.shape, np.nan)
    mean, masks = from_edge_distributions_px(combined_result,bin_width_px = bin_width_px,path=False,save_name = False)
    for i, mask in enumerate(masks):
        radial_matrix[mask] = mean[i]
    return radial_matrix

def process_image_wrapper(read_dir_full_cropped = 'Paintings/Processed/Raw/Full',
                          testing_master = pd.read_parquet('master.parquet'),
                          learner_folder = 'gcp_classifier-3_10-Max_color_10-15-2022',
                          group = 'XP',
                          center_tile = 'both',
                          specific_files_in_master = False, #should be list of files if used
                          verbose = False,
                          add = 0,
                          viz_dir = 'viz',
                          painting_preds_dir = 'painting_preds',
                          specify_viz_save = False,
                          do_tiling = True,
                          item_tfms = False,
                          write_individual_layers = False,
                          do_report_data = True,
                          update = False,
                          resize = (500,500)
                          ):
    if specific_files_in_master:
        assert isinstance(specific_files_in_master,list) , 'needs to be a list if used'
        paintings = specific_files_in_master
    else:
        paintings = testing_master.file.tolist()
    if center_tile == 'both':
        center_tile = True
        repeat = True
    else:
        repeat = False
    if center_tile:
        descriptor = 'standard'
        do_viz = False
    else:
        descriptor = 'overlap'
        do_viz = True,
    # do_viz = False,
    if add > 0:
        descriptor = descriptor+ '_' + str(add)   
    folder = group + '_' + descriptor

    viz_path_start = os.path.join(viz_dir,folder)
    save_pre_path = os.path.join(painting_preds_dir,folder)
    bin_width_cm = 10
    for file_name in paintings:
        if update:
            if not os.path.exists(os.path.join(save_pre_path,file_name+ '_' + descriptor +'_' + learner_folder,'results.csv')):
                process_image(testing_master,
                            file_name,
                            center_tile = center_tile,
                            tile_update = False,
                            start = bin_width_cm,
                            read_dir_full_cropped = read_dir_full_cropped,
                            learner_folder = learner_folder,
                            select_painting = 'P2(V)', #only for comparison if making report
                            inverse_pollock_signatures = True,
                            resize = resize,
                            write_individual_layers=write_individual_layers,
                            viz_path_start = viz_path_start,
                            viz_path_end = descriptor,
                            save_pre_path = save_pre_path,
                            save_folder_description = descriptor,
                            make_report = False,
                            do_tiling = do_tiling,
                            do_report_data = do_report_data,
                            do_viz = do_viz,
                            add = add, 
                            pollock_map_colors = [(1, 0, 0), (0, 0, 0), (0, 0.5, 0)],
                            write_pollock_map  = True,
                            show_figure = False,
                            verbose = verbose,
                            specify_viz_save = specify_viz_save,
                            item_tfms = item_tfms
                            )
        else:
            process_image(testing_master,
                        file_name,
                        center_tile = center_tile,
                        tile_update = False,
                        start = bin_width_cm,
                        read_dir_full_cropped = read_dir_full_cropped,
                        learner_folder = learner_folder,
                        select_painting = 'P2(V)', #only for comparison if making report
                        inverse_pollock_signatures = True,
                        resize = resize,
                        write_individual_layers=write_individual_layers,
                        viz_path_start = viz_path_start,
                        viz_path_end = descriptor,
                        save_pre_path = save_pre_path,
                        save_folder_description = descriptor,
                        make_report = False,
                        do_tiling = do_tiling,
                        do_report_data = do_report_data,
                        do_viz = do_viz,
                        add = add, 
                        pollock_map_colors = [(1, 0, 0), (0, 0, 0), (0, 0.5, 0)],
                        write_pollock_map  = True,
                        show_figure = False,
                        verbose = verbose,
                        specify_viz_save = specify_viz_save,
                        item_tfms = item_tfms
                        )
    if repeat:
        center_tile = True
        process_image_wrapper(read_dir_full_cropped = read_dir_full_cropped,
                            testing_master = testing_master,
                            group = group,
                            center_tile = center_tile,
                            specific_files_in_master =specific_files_in_master, #should be list of files if used
                            verbose = verbose,
                            add = 0,
                            item_tfms = item_tfms,
                            learner_folder = learner_folder,
                            update = True
                            )


def get_combined_list(painting_list,viz_path_start = 'viz/PJ',viz_path_end = 'overlap_resize',threshold=0.56,verbose = False,change_aspect = (500,500),blur = False):
    combined_list = []
    for painting in painting_list:
        util.print_if(verbose,painting)
        viz_path = os.path.join(viz_path_start,painting+'_' +viz_path_end)
        cr = np.load(os.path.join(viz_path,'combined.npy'))
        if blur:
            cr = gaussian_blur(cr)
        if change_aspect:
            combined_list.append(Resize(cr,change_aspect,mode='constant'))
        else:
            combined_list.append(cr)
    return combined_list

def get_combined_radial_list(combined_list,blur = False):
    combined_radial_list = []
    for item in combined_list:
        if blur:
            item = gaussian_blur(item)
        combined_radial_list.append(get_radial_from_combined(item))
    return combined_radial_list