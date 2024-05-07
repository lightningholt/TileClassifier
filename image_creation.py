import numpy as np
from PIL import Image
import os
import random
import cv2
import shutil

def single_color_image(painting_name,width=1000,height=1000,rgb=(0,0,0),path='Paintings\Processed\Raw\Test Imitations'):
    img  = Image.new( mode = "RGB", size = (width, height), color=rgb)
    img.save(os.path.join(path,painting_name + '.tif'))
    img.show()
    return img

def random_pixel_image( painting_name,image_size=(300, 300, 3),color_low=0,color_high=256,path='Paintings\Processed\Raw\Test Imitations'):
    arr = np.random.randint(
    low=color_low, 
    high=color_high,
    size=image_size,
    dtype=np.uint8
    )
    img = Image.fromarray(arr)
    img.save(os.path.join(path,painting_name + '.tif'))
    img.show()
    return img

def make_bars(bar_width = (50,50), image_size = (900,900),white2color=(255, 255, 255), black2color=(0, 0, 0), save=None):
    # Extracting parameters from the input tuples
    num_pixels_black, num_pixels_white = bar_width
    height_pixels, width_pixels = image_size

    # Calculating the total number of bars
    total_bars = num_pixels_black + num_pixels_white

    # Creating the black and white bars pattern
    pattern = [0] * num_pixels_black + [255] * num_pixels_white
    pattern = pattern * (width_pixels // total_bars + 1)

    # Trimming the pattern to the desired image width
    pattern = pattern[:width_pixels]

    # Creating the image with the pattern repeated for the height
    image_data = pattern * height_pixels

    # Creating the image
    image = Image.new('L', image_size)
    image.putdata(image_data)

    image = convert_grayscale_to_color(image,white2color=white2color, black2color=black2color)
    # Saving the image if a path is provided
    if save:
        image.save(save)

    return image

def crop_bars(image, save=None,enlarge_factor = 3/2,white2color=(255, 255, 255), black2color=(0, 0, 0)):
    # Calculate one-third of the width and height to crop from each side
    division_factor = int(enlarge_factor*4)
    crop_width = int(image.width // division_factor)
    crop_height = int(image.height // division_factor)

    # Calculate the crop coordinates
    left = crop_width
    upper = crop_height
    right = image.width - crop_width
    lower = image.height - crop_height

    # Crop the image
    cropped_image = image.crop((left, upper, right, lower))

    # Saving the cropped image if a path is provided
    cropped_image = convert_grayscale_to_color(cropped_image,white2color=white2color, black2color=black2color)
    if save:
        cropped_image.save(save)

    return cropped_image

def rotate_bars(image, angle,white2color=(255, 255, 255), black2color=(0, 0, 0), save=None):
    # Rotating the image
    rotated_image = image.rotate(angle, resample=Image.BICUBIC)
    rotated_image = convert_grayscale_to_color(rotated_image,white2color=white2color, black2color=black2color)
    # Saving the rotated image if a path is provided
    if save:
        rotated_image.save(save)
    return rotated_image
        
def make_bar_image(angle,bar_width = (50,50),image_size= (900,900),white2color=(255, 255, 255), black2color=(0, 0, 0),save = 'TB', path = 'Imitations_Summer_2023/bar_images/',save_end = '_cropped.tif'):
    enlarge_factor = 3/2
    enlarged_size = tuple([int((enlarge_factor)*i) for i in image_size])
    image=make_bars(bar_width=bar_width, image_size=enlarged_size,white2color=white2color, black2color=black2color, save=None)
    rotated_image = rotate_bars(image, angle=angle, white2color=white2color, black2color=black2color, save=None)
    angle_string=str(angle)
    image_size_string=str(image_size[0])
    if save:
        if not os.path.exists(path):
            os.makedirs(path)
        save = os.path.join(path,save+"_" + angle_string + "_" + image_size_string + save_end)
    image=crop_bars(rotated_image,white2color=white2color, black2color=black2color, save=save)
    return image

def bar_image_wrapper(angle_list,bar_width = (50,50),image_size = (900,900),white2color=(255, 255, 255), black2color=(0, 0, 0),save = 'TB', path = 'Imitations_Summer_2023/bar_images/',save_end = '_cropped.tif'):
    for i in angle_list:
       image=make_bar_image(i,bar_width = bar_width,image_size = image_size,white2color=white2color, black2color=black2color,path = path, save = save,save_end = save_end) 
    return image

def convert_grayscale_to_color(img, white2color=(255, 255, 255), black2color=(255,255, 255)):
    if isinstance(img, Image.Image) and (img.mode == 'L'):
        # Create a new RGB image
        rgb_img = Image.new("RGB", img.size)
        
        # Get pixel data from the grayscale image
        grayscale_pixels = img.getdata()
        
        # Convert grayscale pixels to color pixels based on conditions
        color_pixels = []
        for pixel in grayscale_pixels:
            if pixel == 255:  # White pixel
                color_pixels.append(white2color)
            elif pixel == 0:  # Black pixel
                color_pixels.append(black2color)
            else:
                color_pixels.append((pixel,pixel,pixel))  # Other shades of gray
        
        # Put the color pixels into the new RGB image
        rgb_img.putdata(color_pixels)
        
        # Save the color image
        return rgb_img
    else:
        return img

def generate_random_lines_image(width, height, line_width, num_lines = 50,save_tiff=False, line_color = (0, 0, 0) ,tiff_filename='output.tif'):
    background_color = (255, 255, 255)  # White color (RGB)
    # line_color = (0, 0, 0)  # Black color (RGB)

    # Create a white background
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:, :] = background_color

    # Generate random lines
    # num_lines = random.randint(5, 100)
    for _ in range(num_lines):
        start_x = random.randint(0, width)
        start_y = random.randint(0, height)
        end_x = random.randint(0, width)
        end_y = random.randint(0, height)
        cv2.line(img, (start_x, start_y), (end_x, end_y), line_color, line_width)

    # # Apply random rotation
    # angle = random.randint(0, 360)
    # rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
    # img = cv2.warpAffine(img, rotation_matrix, (width, height), borderValue=background_color)

    if save_tiff:
        pil_image = Image.fromarray(img)
        pil_image.save(tiff_filename)

    return img

from PIL import Image, ImageDraw

def create_gradient_image(image_size):
    gradient_image = Image.new("L", image_size, color=0)  # "L" mode represents grayscale image
    draw = ImageDraw.Draw(gradient_image)

    for x in range(image_size[0]):
        # Calculate the grayscale value based on the x-coordinate
        # Linear interpolation from black (0) to white (255)
        grayscale_value = int(x / image_size[0] * 255)
        draw.line((x, 0, x, image_size[1]), fill=grayscale_value)

    return gradient_image

def make_gradient_image (angle,image_size= (900,900),save = 'TG', path = 'Imitations_Summer_2023/bar_images/',save_end = '_cropped.tif'):
    enlarge_factor = 3/2
    enlarged_size = tuple([int((enlarge_factor)*i) for i in image_size])
    image=create_gradient_image(image_size=enlarged_size)
    rotated_image = rotate_bars(image, angle=angle, save=None)
    angle_string=str(angle)
    image_size_string=str(image_size[0])
    if save:
        if not os.path.exists(path):
            os.makedirs(path)
        save = os.path.join(path,save+"_" + angle_string + "_" + image_size_string + save_end)
    image=crop_bars(rotated_image, save=save)
    return image

# def rotate_bars(image, angle, save=None):
#     # Rotating the image
#     rotated_image = image.rotate(angle, resample=Image.BICUBIC)
#     # Saving the rotated image if a path is provided
#     if save:
#         rotated_image.save(save)
#     return rotated_image

def gradient_image_wrapper(angle_list,image_size= (900,900),save = 'TG', path = 'Imitations_Summer_2023/bar_images/',save_end = '_cropped.tif'):
    for i in angle_list:
       image=make_gradient_image(i,image_size = image_size,path = path, save = save,save_end = save_end) 
    return image

# def crop_bars(image, save=None):
#     # Calculate one-third of the width and height to crop from each side
#     crop_width = image.width // 6
#     crop_height = image.height // 6

#     # Calculate the crop coordinates
#     left = crop_width
#     upper = crop_height
#     right = image.width - crop_width
#     lower = image.height - crop_height

#     # Crop the image
#     cropped_image = image.crop((left, upper, right, lower))

#     # Saving the cropped image if a path is provided
#     if save:
#         cropped_image.save(save)

#     return cropped_image

def create_image_copies(size,parent,folder,image_pathway,new_image_pathway):
    files = os.listdir(os.path.join(parent,folder))
    for i in range(len(files)):
     
        image_path = "" + image_pathway + "" + files[i] + "" 

        new_image_name = "" + files[i] + "_" + size +"CM.tiff"

        new_folder_path = new_image_pathway
        
        new_image_path = copy_and_rename_image(image_path, new_folder_path, new_image_name)

        print(f"New image path: {new_image_path}") 
        print(new_image_name)
        
def copy_and_rename_image(image_path, new_folder_path, new_image_name):
    # Create the new folder if it doesn't exist
    os.makedirs(new_folder_path, exist_ok=True)

    # Construct the path for the copied and renamed image
    new_image_path = os.path.join(new_folder_path, new_image_name)

    # Copy and rename the image
    shutil.copy(image_path, new_image_path)

    return new_image_path
