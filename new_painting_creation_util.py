def make_bars(bar_width, image_size, save=None):
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

    # Saving the image if a path is provided
    if save:
        image.save(save)

    return image

def crop_bars(image, save=None):
    # Calculate one-third of the width and height to crop from each side
    crop_width = image.width // 6
    crop_height = image.height // 6

    # Calculate the crop coordinates
    left = crop_width
    upper = crop_height
    right = image.width - crop_width
    lower = image.height - crop_height

    # Crop the image
    cropped_image = image.crop((left, upper, right, lower))

    # Saving the cropped image if a path is provided
    if save:
        cropped_image.save(save)

    return cropped_image

def rotate_bars(image, angle, save=None):
    # Rotating the image
    rotated_image = image.rotate(angle, resample=Image.BICUBIC)

    # Saving the rotated image if a path is provided
    if save:
        rotated_image.save(save)
    return rotated_image
        
def make_bar_image (angle,bar_width,image_size):
    image=make_bars(bar_width=bar_width, image_size=image_size, save=None)
    rotated_image = rotate_bars(image, angle=angle, save=None)
    angle_string=str(angle)
    image_size_string=image_size[0]*2 // 3
    image_size_string=str(image_size_string)
    image=crop_bars(rotated_image, save="Imitations_Summer_2023/bar_images/TB_" + angle_string + "_" + image_size_string + ".tiff")
    return image



    return rotated_image

def bar_image_wrapper(angle_list,bar_width,image_size):
    for i in angle_list:
       image=make_bar_image (i,bar_width,image_size) 
    return image

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

def make_gradient_image (angle,image_size):
    image=create_gradient_image(image_size=image_size)
    rotated_image = rotate_bars(image, angle=angle, save=None)
    angle_string=str(angle)
    image_size_string=image_size[0]*2 // 3
    image_size_string=str(image_size_string)
    image=crop_bars(rotated_image, save="Imitations_Summer_2023/gradient_images/GI_" + angle_string + "_" + image_size_string + ".tiff")
    return image

def rotate_bars(image, angle, save=None):
    # Rotating the image
    rotated_image = image.rotate(angle, resample=Image.BICUBIC)

    # Saving the rotated image if a path is provided
    if save:
        rotated_image.save(save)
    return rotated_image

def gradient_image_wrapper(angle_list,image_size):
    for i in angle_list:
       image=make_gradient_image (i,image_size) 
    return image

def crop_bars(image, save=None):
    # Calculate one-third of the width and height to crop from each side
    crop_width = image.width // 6
    crop_height = image.height // 6

    # Calculate the crop coordinates
    left = crop_width
    upper = crop_height
    right = image.width - crop_width
    lower = image.height - crop_height

    # Crop the image
    cropped_image = image.crop((left, upper, right, lower))

    # Saving the cropped image if a path is provided
    if save:
        cropped_image.save(save)

    return cropped_image

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

