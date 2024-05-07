import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image, ImageOps
from util import get_rows_columns_from_files
import util
import utils_fastai
import pandas as pd
from pathlib import Path
import seaborn as sns
import plotting
import matplotlib.lines as mlines
import visualizer as vz
import cv2
import rubric

def make_slice_row_fig(name = 'A66',slice_sizes=['5','25','45','Max'],folder = 'Paintings/Processed/Raw/',figsize=(12, 4), dpi=80,save=False,selected_thumbnails = False,selected_paths = False):
    if selected_thumbnails:
        assert len(selected_thumbnails)==len(slice_sizes), 'slice_sizes and selected_thumbnails must be same length'
        thumbnails = selected_thumbnails
    else:
        thumbnails = []

    sns.set_style(style=None, rc=None)
    fig = plt.figure(figsize=figsize,facecolor='white')

    for i,slice_size in enumerate(slice_sizes):
        if selected_paths:
            im = Image.open(selected_paths[i])
        else:
            path = os.path.join(folder,str(slice_size))
            # print(path,name)
            files = [item for item in os.listdir(path) if item.startswith(name+'_c')]
            # print(slice_size,files)
            file_start = files[0].split('_C')[0] + '_'
            file_end = '.' + files[0].split('.')[1]
            # print(files[0])
            if selected_thumbnails:
                file = file_start+selected_thumbnails[i]+file_end
                im = Image.open(os.path.join(path,file))
            else:
                file = files[random.randint(0, len(files)-1)]
                thumbnails.append('C'+file.split('_C')[1].split('.')[0])
                im = Image.open(os.path.join(path,file))
        ax = fig.add_subplot(1, len(slice_sizes), i+1)
        plt.imshow(im)
        plt.tick_params(axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,
            left = False, # ticks along the top edge are off
            labelbottom=False)
        ax.set_yticklabels([])
        if str(slice_size).isnumeric():
            ax.set_title(str(slice_size)+' cm',fontsize = 18)
        else:
            ax.set_title(str(slice_size),fontsize = 18)
    plt.show()
    if save:
        fig.savefig(save, dpi=dpi)
    return fig
def get_slice_row_paths(name = 'A66',slice_sizes=['5','25','45','Max'],folder = 'Paintings/Processed/Raw/',selected_thumbnails = False,selected_paths = False):
    if selected_thumbnails:
        assert len(selected_thumbnails)==len(slice_sizes), 'slice_sizes and selected_thumbnails must be same length'
        thumbnails = selected_thumbnails
    else:
        thumbnails = []

    paths = []
    for i,slice_size in enumerate(slice_sizes):
        if selected_paths:
            paths = selected_paths
        else:
            path = os.path.join(folder,str(slice_size))
            # print(path,name)
            files = [item for item in os.listdir(path) if item.startswith(name+'_c')]
            # print(slice_size,files)
            file_start = files[0].split('_C')[0] + '_'
            file_end = '.' + files[0].split('.')[1]
            # print(files[0])
            if selected_thumbnails:
                file = file_start+selected_thumbnails[i]+file_end
                paths.append(os.path.join(path,file))
            else:
                file = files[random.randint(0, len(files)-1)]
                thumbnails.append('C'+file.split('_C')[1].split('.')[0])
                paths.append(os.path.join(path,file))
    return paths

def plot_slice_grid(name = 'A66',folder = 'Paintings/Processed/Raw/' ,slice_size = '10',figsize=20,axes_pad=0.1,save=False,dpi=80,show = True):
    path = os.path.join(folder,str(slice_size))
    files = [item for item in os.listdir(path) if item.startswith(name+'_c')]
    num_rows,num_columns,rows,columns = get_rows_columns_from_files(files)

    img_arr = []

    files = sorted(files)
    for file in files:
        img = Image.open(os.path.join(path,file))
        img_arr.append(img)

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np

    if isinstance(figsize,int):
        figsize = (figsize*num_rows,figsize*num_columns)
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(num_columns, num_rows),  # creates 2x2 grid of axes
                     axes_pad=axes_pad,  # pad between axes
                     )

    for ax, im in zip(grid, img_arr):
        ax.imshow(im)
        ax.tick_params(axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,
            left = False, # ticks along the top edge are off
            labelbottom=False)
        ax.set_yticklabels([])
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

def plot_problem_paintings(folder,save=False,cutoff = 0.7):
    results= pd.read_csv(Path(folder,'results.csv'), low_memory=False)
    r = util.vote_system(results,one_vote=False, binarize=False, binarize_prob=0.5, decision_prob=0.50)
    r = util.get_accuracy_column(r,sort = 'accuracy')
    if not os.path.exists(save):
        os.mkdir(save)
    # print(len(utils_fastai.result_paintings(results,category=tuple(r[r.accuracy < cutoff].painting.tolist()))))
    for painting in utils_fastai.result_paintings(results,category=tuple(r[r.accuracy < cutoff].painting.tolist())):
        # print(painting)
        if not '.' in painting: #sloppy work around
            utils_fastai.plot_slice_acc(painting,results,save = Path(save, painting + '.png'))


def make_cat_df(res_df, OneVote, cat_dict=None):
    if cat_dict is None:
        # map groups to P = Pollock, A = Abstract Drips, N = Non-drips
        cat_dict = {'P':'P',
                    'J':'P',
                    'A':'A',
                    'C':'A',
                    'D':'A',
                    'E':'A',
                    'G':'A',
                    'F':'N'}

    cat_df = util.vote_system(res_df, one_vote=OneVote, binarize=False)
    cat_df['group'] = cat_df['painting'].str[0]
    cat_df['category'] = cat_df['group'].map(cat_dict)
    cat_df.set_index(['category', 'painting'], inplace=True)

    return cat_df


def polar_to_cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def convert_deg_to_rads(angle_in_degrees):
    return angle_in_degrees * (np.pi / 180)

def convert_rad_to_degs(angle_in_rads):
    return angle_in_rads * (180/np.pi )

def place_radii_in_annulus(data_length, bottom, max_height_from_bottom,seed = False):
    if seed:
        np.random.seed(seed) 
    radii = np.random.rand(data_length)
    radii = radii * max_height_from_bottom + bottom
    return radii


def data_to_angle(data, start_angle, full_angle):
    theta = full_angle * (1 - data) + start_angle
    return theta


def add_arrow(ax, length, angle, start_pos=(0,0), lw=0.001, color='gray', zz=3, aa=1, ls='-',head_length=0, head_width=0):
    if angle > 2*np.pi:
        angle = convert_deg_to_rads(angle)
    # angle = convert_deg_to_rads(angle)
    arrow_x, arrow_y = polar_to_cart(length, angle)
    if isinstance(start_pos,(float,int)):
        start_pos = polar_to_cart(start_pos, angle)
    ax.arrow(start_pos[0], start_pos[1], arrow_x - start_pos[0], arrow_y - start_pos[1],
                width=lw, head_width=head_width, head_length=head_length, fc=color, ec=color,
                zorder=zz, alpha=aa, linestyle=ls)    
    return ax


def add_bound_circle(ax, radius, angles, color='k', ls='-', a=1):
    ax.plot(radius * np.cos(angles), radius * np.sin(angles),
            color=color, linestyle=ls, alpha=a)
    return ax


def find_comparators(cat_df, prob_to_test):
    cats = cat_df.index.get_level_values(0).unique()
    # print(cats)
    comp_names = []

    for cat in cats:
        name = (cat_df.loc[cat, 'pollock_prob'] - prob_to_test).abs().idxmin()
        comp_names.append(name)
    return comp_names


def shrink_paintings(im, area=40):
    height, width = im.shape[:2]

    aspect_ratio = height / width

    new_width = np.sqrt(area / aspect_ratio)
    new_height = np.sqrt(aspect_ratio * area)

    return new_width, new_height


def transform_paintings(im, ax, rad, angle, area=40):
    dial_x, dial_y = polar_to_cart(rad, angle)

    paint_x, paint_y = shrink_paintings(im, area=area)

    if angle < 0:
        start_y = dial_y - paint_y
        start_x = dial_x# + paint_x
        anch = 'NW'
    elif angle <= np.pi/2:
        start_x = dial_x# + paint_x
        start_y = dial_y
        anch = 'SW'
    elif angle < np.pi:
        start_x = dial_x - paint_x
        start_y = dial_y
        anch = 'SE'
    else:
        start_y = dial_y - paint_y
        start_x = dial_x - paint_x
        anch='NE'
    # print(start_x, start_y)

    paint_coords = [start_x, start_y, paint_x, paint_y]

    new_ax = ax.inset_axes(paint_coords,
                           zorder=5, transform=ax.transData, anchor=anch)

    return new_ax, paint_coords


def add_comparator_image(ax, name, p_angle, x, y, line_length, area=40, color='k',
              path='Paintings/Processed/Descreened/Full/', ext='_cropped_descreened.tif'):

    ax = add_arrow(ax, line_length, p_angle, start_pos=(x,y), color=color, zz=-1, ls='dashed', aa=0.5)

    im = plt.imread(path + name + ext)
    paint_ax, coords = transform_paintings(im, ax, line_length,
                                   p_angle, area=area)
    paint_ax.axis('off')
    paint_ax.imshow(im)
    return paint_ax, coords


#outdated
def dial_figure(test_img_name_plus_path, test_img_prob, res_df,
 path_to_images='Paintings/Processed/Descreened/Full/',
 ext='_cropped_descreened.tif', OneVote=False,
 save_name=None):

    if 'Unnamed: 0' in res_df.columns:
        res_df.drop('Unnamed: 0', axis=1, inplace=True)

    cat_df = make_cat_df(res_df, OneVote, cat_dict=None)

    # pre-amble plot stuff
    fig_size_length = 10
    max_height = 2  # max_height probably should have been called max radius = max distance above the colored circle
    start_height = 10  # probably should have been called start_radius
    total_height = start_height + max_height
    full_angle = 3*np.pi/2
    embed_area = 4 * start_height
    large_font_size = 20

    start_angle = -convert_deg_to_rads(45)
    buffer_width_start = convert_deg_to_rads(1.25)
    buffer_width_end = convert_deg_to_rads(1.5)
    bound_angles = np.linspace(start_angle - buffer_width_start,
                           full_angle + start_angle + buffer_width_end,
                           100)
    #describe data set
    pick_a_cat = ['P', 'A', 'N']
    categories = ['Pollock', 'Non-Pollock Poured', 'Non-Pollock Abstract']
    print('categories are', pick_a_cat)
    colors = ['g', 'b', 'r', 'k']

    # find extreme performing comparators
    # (paintings with pollock_prob closest to 1 or 0)
    extreme_ones = find_comparators(cat_df, 1)
    extreme_zeros = find_comparators(cat_df, 0)
    extremes = extreme_ones + extreme_zeros

    fig, ax = plt.subplots(1,1, figsize=(fig_size_length, fig_size_length),
                            constrained_layout=True)
    ax.axis('equal')
    ax.axis('off')

    # add outer bounding circle and limits
    # ax = add_bound_circle(ax, radius=total_height,
    #                   angles=bound_angles, color='gray')
    # ax = add_arrow(ax, max_height+start_height, start_angle - buffer_width_start)
    # ax = add_arrow(ax, max_height + start_height,
    #                full_angle + start_angle + buffer_width_end)

    # add bounding circle
    ax = add_bound_circle(ax, radius=start_height,
                      angles=bound_angles, color='gray', ls='-')
    ax = add_arrow(ax, start_height, start_angle - buffer_width_start)
    ax = add_arrow(ax, start_height, full_angle + start_angle + buffer_width_end)
    # add dial marker labels
    number_of_numbers = 5
    dial_dict = {'size': 3 * large_font_size/4}
    text_thetas = np.linspace(full_angle + start_angle, start_angle, number_of_numbers)
    dial_numbers_text = [f'{ii/100:0.2f}' for ii in np.linspace(0, 100, number_of_numbers)]
    rad_thetas = start_height + 0.5 * max_height * np.ones(len(text_thetas))
    dial_numbers_x, dial_numbers_y = polar_to_cart(rad_thetas, text_thetas)
    for tex in np.arange(number_of_numbers):
        ax.text(dial_numbers_x[tex], dial_numbers_y[tex], dial_numbers_text[tex], ha="center", va="center", fontdict=dial_dict)
        if tex not in [0, number_of_numbers-1]:
            start_arr_dial = polar_to_cart(start_height - max_height * len(pick_a_cat), text_thetas[tex])
            ax = add_arrow(ax, start_height, text_thetas[tex], start_pos=start_arr_dial, ls='dotted', color='gray')


    # get dataframe of coords to space images out with
    img_coords = pd.DataFrame(index= np.arange(2*len(pick_a_cat)),
                          columns=['Angle', 'x_start', 'y_start', 'ext_x', 'ext_y'])
    coord_idx = 0

    #iterate through categories
    for ii in np.arange(len(pick_a_cat)):
        bottom = start_height - max_height * (ii+1)

        # possibly plot all slices for test image (add TEST to categories)
        if len(pick_a_cat[ii]) == 1:
            data = cat_df.loc[pick_a_cat[ii], 'pollock_prob']
        else:
            data = res.loc[res['painting_name'] == name, 'pollock_prob']

        thetas = data_to_angle(data, start_angle, full_angle)
        rads = place_radii_in_annulus(len(thetas), bottom + 0.05 * max_height,
                                        0.8 * max_height)
        x,y = polar_to_cart(rads, thetas)

        # scatter category and add bottom radius
        c = ax.scatter(x, y, c=colors[ii], alpha=0.5, s=24, label=categories[ii])
        ax = add_bound_circle(ax, radius=bottom, angles=bound_angles,
                                color=colors[ii], ls='--', a=0.5)

        # add images of extreme examples
        subset_paints = [pp for pp in extremes if pp in data.index]
        for painting in subset_paints:

            paint_angle = thetas.loc[painting]
            if paint_angle in img_coords['Angle'].unique():
                line_length = 1.8 * total_height
            else:
                line_length = 1.1 * total_height
            paint_ax, coords = add_comparator_image(ax, painting, paint_angle,
                                               x.loc[painting], y.loc[painting],
                                               line_length = line_length,
                                               area = embed_area,
                                               color = colors[ii],
                                               path = path_to_images, ext = ext)
            img_coords.loc[coord_idx, :] = [thetas.loc[painting]] + coords
            coord_idx += 1

    leg_x, leg_y = polar_to_cart(total_height, full_angle)
    ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y))

    data_angle = data_to_angle(test_img_prob, start_angle, full_angle)
    if data_angle in img_coords['Angle'].unique():
        pointer_line = total_height
    else:
        pointer_line = 1.1 * total_height

    ax = add_arrow(ax, pointer_line, data_angle, color='k', lw=0.1, zz=-1)

    im = plt.imread(test_img_name_plus_path)
    final_ax, coords = transform_paintings(im, ax, pointer_line, data_angle,
                                            area=embed_area)
    final_ax.axis('off')
    final_ax.imshow(im)

    text_x, text_y = polar_to_cart(start_height - 2*max_height, full_angle)
    font = {'fontname': 'Arial',
            'weight': 'bold',
            'size': large_font_size,
            }
    ax.text(text_x, text_y, f'PMF: {test_img_prob:.02f}', fontdict=font, ha="center", va="center")

    if save_name is not None:
        if len(save_name.split('.')) == 1:
            save_name += '.png'
        plt.savefig(save_name)

    plt.show()

    return img_coords

def find_rows_with_color(pixels, width, height, color):
    rows_found=[]
    for y in range(height):
        for x in range(width):
            if pixels[x, y] != color:
                break
        else:
            rows_found.append(y)
    return rows_found
def remove_vertical_white_padding(path,save=False,location = 0):
    old_im = Image.open(path)
    if old_im.mode != 'RGB':
        old_im = old_im.convert('RGB')
    pixels = old_im.load()
    width, height = old_im.size[0], old_im.size[1]
    rows_to_remove = find_rows_with_color(pixels, width, height, (255, 255, 255))
    rows_to_remove = util.get_sequences(rows_to_remove)
    rows_to_remove = rows_to_remove[int(location)]
    new_im = Image.new('RGB', (width, height - len(rows_to_remove)))
    pixels_new = new_im.load()
    rows_removed = 0
    for y in range(old_im.size[1]):
        if y not in rows_to_remove:
            for x in range(new_im.size[0]):
                pixels_new[x, y - rows_removed] = pixels[x, y]
        else:
            rows_removed += 1
    if save:
        new_im.save(save)

def plot_slice_hist(painting, results, error_bars=False, save=False, title=True, x_label='Tile Size[cm]', y_label='Pollock Signature', dpi=80, one_vote=False, num_colors=100, c_range=0.7,
                    colors=[(1, 0, 0), (0, 0, 0), (0, 0.5, 0)],
                    selected_paths = False,
                    slice_sizes = False,
                    threshold = 0.56):
    acc_by_group, painting_acc_at_sizes, error_group,error_painting = utils_fastai.painting_acc_by_slice_size(results,prob = True)
    slice_size_list = [str(item) for item in results.slice_size.tolist()]
    min_size = min([int(item) for item in slice_size_list if item.isnumeric()])
    painting_acc_index = [str(item) for item in painting_acc_at_sizes[painting].index]
    sizes = [int(item) for item in painting_acc_index if item.isdigit()]
    keys = [str(item) for item in range(min_size,np.max(sizes)+5,5)] + ['Max']
    data = {key:painting_acc_at_sizes[painting][key] for key in keys}
    # print(data)

    data= pd.Series(data)
    cmap = vz.get_custom_colormap(colors =  colors)
        
    # Set up GridSpec for the figure
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    gs = fig.add_gridspec(3, 6, height_ratios=[0.5,0, 1], width_ratios=[0.03, 0.13, .22,.22,.22,.22])

    # Add the main plot (ax)
    ax = fig.add_subplot(gs[2, 2:], facecolor='white')
    bars = ax.bar(data.index, data, color=cmap(data))
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    if title:
        ax.set_title(painting + ' ' + y_label, fontsize=24)

    ax.set_ylim(0, 1)
    if one_vote:
        arrowprops = dict(arrowstyle='->',linewidth=2.5,color = 'k')
        ax.axhline(np.mean(data), linestyle='dashed', c='black')
        ax.annotate('PMF', xy=(1, np.mean(data)),xytext =(1.07,np.mean(data)), xycoords=('axes fraction', 'data'), ha='left', va='center', color='black', fontsize=16,
                    arrowprops = arrowprops)

    if len(data) > 25:
        for label in ax.xaxis.get_ticklabels()[::-1][1::2][::-1]:
            label.set_visible(False)
    

    ax.set_xlim(-1, len(data.index))
    ax.set_xticks(data.index)  # Ensure all x-ticks are displayed
    ax.set_xticklabels([str(i) for i in data.index],rotation = 90)

    # Thumbnail axis (ax_thumbs)
    if selected_paths:
        ax_thumbs = []
        for i in range(4):
            ax_thumbs.append(fig.add_subplot(gs[0, i+2], facecolor='white'))
            ax_thumbs[i].axis('off')
            im = Image.open(selected_paths[i])
            ax_thumbs[i].imshow(im)
            if slice_sizes:
                ax_thumbs[i].set_title(slice_sizes[i])

    # Add content to the ax_thumbs here (if needed)

    # Colorbar axis (cax)
    cax = fig.add_subplot(gs[2, 0])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax)
    custom_ticks = [0, threshold, 1]  # Replace with your desired custom tick positions
    cbar.set_ticks(custom_ticks)
    cbar.ax.yaxis.set_ticks_position('left')
    # cbar.set_label('Data Values')

    plt.subplots_adjust(bottom=0.3, left=0.2, hspace=0, wspace=0) 

    # plt.xticks(rotation=45)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')

    plt.show()

# def plot_slice_hist(painting, results,error_bars = False,save = False,title = True,x_label = 'Tile Size[cm]',y_label = 'Pollock Signature',dpi=80,one_vote = False,num_colors=100,c_range=0.7,
#                     colors = [(1, 0, 0), (0, 0, 0), (0, 0.5, 0)]):
#     acc_by_group, painting_acc_at_sizes, error_group,error_painting = utils_fastai.painting_acc_by_slice_size(results,prob = True)
#     slice_size_list = [str(item) for item in results.slice_size.tolist()]
#     min_size = min([int(item) for item in slice_size_list if item.isnumeric()])
#     painting_acc_index = [str(item) for item in painting_acc_at_sizes[painting].index]
#     sizes = [int(item) for item in painting_acc_index if item.isdigit()]
#     keys = [str(item) for item in range(min_size,np.max(sizes)+5,5)] + ['Max']
#     data = {key:painting_acc_at_sizes[painting][key] for key in keys}
#     # print(data)

#     data= pd.Series(data)
#     if colors:
#         cmap = vz.get_custom_colormap(colors =  colors)
#         values = np.linspace(0, 1, num_colors+1)
#         sns_colors = plt.cm.get_cmap(cmap)(values)[:, :3]
#         pal = sns.color_palette(sns_colors)
#     else:
#         pal = sns.color_palette("Greens_d", num_colors+1)
#     # pal.reverse()
#     # # rank = data.argsort().argsort()  # http://stackoverflow.com/a/6266510/1628638
#     # rank = round(c_range*num_colors*data).astype(int)
#     # ax = sns.barplot(x=data.index, y=data, palette=np.array(pal[::-1])[rank])

#     fig = plt.figure(facecolor='white')
#     ax = fig.add_subplot(111)
#     bars = ax.bar(data.index, data, color=cmap(data))
#     cax = fig.add_axes([-0.03, 0.3, 0.03, 0.59])
#     cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap),cax=cax)
#     custom_ticks = [0,0.56,1]  # Replace with your desired custom tick positions
#     cbar.set_ticks(custom_ticks)
#     cbar.ax.yaxis.set_ticks_position('left')
#     # cbar.set_label('Data Values')

#     ax_thumbs = fig.add_axes([0, 1, 1, 0.25])

#     ax.set_xlabel(x_label,fontsize = 16)
#     ax.set_ylabel(y_label,fontsize = 16)

#     if title:
#         ax.set_title(painting + ' ' +y_label, fontsize = 24)
#     ax.set_ylim(0, 1)
#     if one_vote:
#         ax.axhline(np.mean(data),linestyle='dashed', c = 'black')
#         ax.annotate(' PMF',xy=(1,np.mean(data)),xycoords=('axes fraction','data'),ha='left',va='center',color='black',fontsize = 16)
#     if len(data) > 25:
#         for label in ax.xaxis.get_ticklabels()[1::2]:
#             label.set_visible(False)
#     # plt.xticks(fontsize=14, rotation=45)
#     # plt.yticks(fontsize=14)
#     plt.subplots_adjust(bottom=0.3)
#     plt.xticks(rotation=45)
#     plt.show()
#     if save:
#         # fig = ax.get_figure()
#         fig.savefig(save, dpi=dpi, bbox_inches='tight')


#outdated
def clean_dial(test_img_name_plus_path, test_img_prob, res_df,
               path_to_images='Paintings/Processed/Descreened/Full/',
               ext='_cropped_descreened.tif', OneVote=False, save_name=None,legend=False,
               dial_markers = True):

    cat_dict = {'P':'P',
                'J':'P',
                'A':'A',
                'C':'A',
                'D':'A',
                'E':'A',
                'G':'A',
                'F':'A'}
    cat_df = make_cat_df(res_df, OneVote, cat_dict=cat_dict)


    # pre-amble plot stuff
    fig_size_length = 10 #6.5
    max_height = 2.7  # max_height probably should have been called max radius = max distance above the colored circle
    start_height = 10  # probably should have been called start_radius
    total_height = start_height + max_height
    full_angle = 3*np.pi/2
    embed_length = 0.75 * start_height
    embed_area = 4 * start_height
    large_font_size = 20


    start_angle = -convert_deg_to_rads(45)
    buffer_width_start = convert_deg_to_rads(1.25)
    buffer_width_end = convert_deg_to_rads(1.5)
    bound_angles = np.linspace(start_angle - buffer_width_start, full_angle + start_angle + buffer_width_end, 100)

    #describe data set
    pick_a_cat = ['P', 'A']#, 'N']
    categories = ['Pollock', 'Non-Pollock']# Poured', 'Non-Pollock Abstract']
    print('categories are', pick_a_cat)
    colors = ['g', 'r', 'r', 'k']

    # find extreme performing comparators
    # (paintings with pollock_prob closest to 1 or 0)
    extreme_ones = find_comparators(cat_df, 1)
    extreme_zeros = find_comparators(cat_df, 0)
    extremes = extreme_ones + extreme_zeros


    fig, ax = plt.subplots(1,1, figsize=(fig_size_length, fig_size_length),
                        constrained_layout=True)
    ax.axis('equal')
    ax.axis('off')

    ax = add_bound_circle(ax, radius=start_height,
                      angles=bound_angles, color='gray', ls='-')
    ax = add_arrow(ax, start_height, start_angle - buffer_width_start)
    ax = add_arrow(ax, start_height, full_angle + start_angle + buffer_width_end)

    # add dial marker labels
    number_of_numbers = 5
    dial_dict = {'size': 3 * large_font_size/4}
    text_thetas = np.linspace(full_angle + start_angle, start_angle, number_of_numbers)
    dial_numbers_text = [f'{ii/100:0.2f}' for ii in np.linspace(0, 100, number_of_numbers)]
    rad_thetas = start_height + 0.5 * max_height * np.ones(len(text_thetas))
    dial_numbers_x, dial_numbers_y = polar_to_cart(rad_thetas, text_thetas)
    for tex in np.arange(number_of_numbers):
        if dial_markers:
            ax.text(dial_numbers_x[tex], dial_numbers_y[tex], dial_numbers_text[tex], ha="center", va="center", fontdict=dial_dict)
        if tex not in [0, number_of_numbers-1]:
            start_arr_dial = polar_to_cart(start_height - max_height * len(pick_a_cat), text_thetas[tex])
            ax = add_arrow(ax, start_height, text_thetas[tex], start_pos=start_arr_dial, ls='dotted', color='gray')

    imp_xy = pd.DataFrame(index=extremes, columns=['x', 'y'])


    for ii in np.arange(len(pick_a_cat)):
        bottom = start_height - max_height * (ii+1)

        # possibly plot all slices for test image (add TEST to categories)
        if len(pick_a_cat[ii]) == 1:
            data = cat_df.loc[pick_a_cat[ii], 'pollock_prob']
        else:
            data = res.loc[res['painting_name'] == name, 'pollock_prob']

        thetas = data_to_angle(data, start_angle, full_angle)
        rads = place_radii_in_annulus(len(thetas), bottom + 0.05 * max_height,
                                        0.8 * max_height)
        x, y = polar_to_cart(rads, thetas)

        selected_paintings = [ii for ii in extremes if ii in data.index]
        imp_xy.loc[selected_paintings, 'x'] = x.loc[selected_paintings]
        imp_xy.loc[selected_paintings, 'y'] = y.loc[selected_paintings]

        # scatter category and add bottom radius
        c = ax.scatter(x, y, c=colors[ii], alpha=0.5, s=24, label=categories[ii])
        ax = add_bound_circle(ax, radius=bottom, angles=bound_angles,
                                color=colors[ii], ls='--', a=0.5)
    if legend:
        ax.legend(loc='lower center', fontsize='x-large')
    # display test
    text_x, text_y = polar_to_cart(start_height - 2*max_height, full_angle)
    font = {'fontname': 'Arial',
            'weight': 'bold',
            'size': large_font_size,
            }
    ax.text(text_x, text_y, f'PMF: {test_img_prob:.02f}', fontdict=font, ha="center", va="center")

    # add pointer to test figure
    data_angle = data_to_angle(test_img_prob, start_angle, full_angle)
    pointer_line = total_height

    ax = add_arrow(ax, pointer_line, data_angle, color='k', lw=0.1, zz=-1)

    # save stuff
    if save_name is not None:
        if len(save_name.split('.')) == 1:
            save_name += '.png'
        plt.savefig(save_name)

    plt.show()

    return extremes


def make_inset_files_for_dial(comparators, path_to_test_img, save_pre_path, save_folder, path_to_comparators='Paintings/Processed/Descreened/Full/', ext='_cropped_descreened.tif'):

    dial_figs = 'dial_figs'

    if not os.path.exists(Path(save_pre_path,save_folder, dial_figs)):
        os.mkdir(Path(save_pre_path,save_folder, dial_figs))

    img =Image.open(path_to_test_img)
    img = ImageOps.contain(img, (2048,2048))
    img.save(Path(save_pre_path,save_folder, dial_figs, 'analyzed_image.png'))


    for painting in comparators:
        img = Image.open(path_to_comparators + painting + ext)
        img = ImageOps.contain(img, (2048,2048))
        img.save(Path(save_pre_path, save_folder, dial_figs, painting+'_image.png'))



def make_row_hist_combined(painting,img_path,learner_results,save_pre_path='',hist_save_name='hist_sim_P.png',save_folder='',one_vote = True):
    #get similar sized painting figures
    y_label = 'Pollock Signature'
    # print(painting)
    slice_sizes = util.get_slice_sizes(learner_results[learner_results.file.str.startswith(painting+'_c')])
    # make_slice_row_fig(name = painting,slice_sizes=slice_sizes,folder = img_path,figsize=(12, 4), dpi=80,save=row_save_sim_P,selected_thumbnails = False)
    paths = get_slice_row_paths(name = painting,slice_sizes=slice_sizes,folder = img_path,selected_thumbnails = False)
    # remove_vertical_white_padding(row_save_sim_P,save=row_save_sim_P,location =-1)
    hist_save_sim_P = Path(save_pre_path,save_folder,hist_save_name)
    plot_slice_hist(painting,learner_results,
                    save=hist_save_sim_P,
                    y_label = y_label,
                    title = False,
                    dpi = 250,
                    one_vote = one_vote,
                    selected_paths = paths,
                    slice_sizes=slice_sizes)
    # remove_vertical_white_padding(hist_save_sim_P,save=hist_save_sim_P,location = 0)


def plot_year_vs_PMF(full,save = False):
    # Calculate average, standard deviation of the mean, and counts
    average_pollock_prob = full.groupby('year')['pollock_prob'].mean()
    pollock_prob_error = full.groupby('year')['pollock_prob'].std() / full.groupby('year')['pollock_prob'].count()**0.5
    year_counts = full['year'].value_counts().sort_index()

    # Plotting the data with error bars and counts
    fig, ax1 = plt.subplots()

    # Plot average and error bars
    ax1.errorbar(average_pollock_prob.index, average_pollock_prob.values, yerr=pollock_prob_error.values, label='Average Pollock Probability')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('PMF')
    ax1.set_ylim([0.5, 1.05])

    # full.boxplot(column='pollock_prob', by='year', ax=ax1)

    # Create a twin axis for counts
    ax2 = ax1.twinx()
    ax2.plot(year_counts.index, year_counts.values, color='red', linestyle='--', label='Counts')
    ax2.set_ylabel('Counts')

    # Combine the legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    # Set the title
    if save:
        plt.title(save)
        plt.savefig(save + '.png')
    plt.show()


def plot_year_vs_PMF_box(full,save = False):
    # Create box and whisker plots
    fig, ax = plt.subplots()
    full.boxplot(column='pollock_prob', by='year', ax=ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('PMF')
    ax.set_ylim([0.5, 1.05])

    # Set the title
    if save:
        plt.title(save)
        plt.savefig(save + '.png')
    plt.show()

def plot_year_vs_PMF_combined(full,save = False):
    df = full.copy()

    # Group the data by 'year' and calculate the values for the box plot
    grouped_data = df.groupby('year')['pollock_prob'].apply(list).reset_index(name='values')

    # Create a box plot
    fig, ax1 = plt.subplots()
    boxplot = ax1.boxplot(grouped_data['values'])
    # ax1.scatter(list(df.year),list(df.pollock_prob))

    # Set x-axis labels as years
    ax1.set_xticklabels(grouped_data['year'])

    # Set labels and title
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Pollock Probability')
    # ax1.set_title('Box Plot of Pollock Probability by Year')
    ax1.set_ylim([0.5, 1.05])

    # Calculate the mean of each box
    box_means = [np.mean(box.get_ydata()) for box in boxplot['medians']]

    # # Plot the line connecting the means
    line = ax1.plot(range(1, len(grouped_data) + 1), box_means, marker='o', color='red')

    # Calculate average, standard deviation of the mean, and counts
    # average_pollock_prob = df.groupby('year')['pollock_prob'].mean()
    # pollock_prob_error = df.groupby('year')['pollock_prob'].std() / df.groupby('year')['pollock_prob'].count()**0.5
    year_counts = df['year'].value_counts().sort_index()

    # Create a twin axis for counts
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(grouped_data) + 1), year_counts.values, color='blue', linestyle='--', label='Counts')
    ax2.set_ylabel('Counts')
    ax2.set_ylim(np.min(year_counts.values)-1,np.max(year_counts.values)+1)

    # Combine the legends from both axes
    # lines = [boxplot['boxes'][0]]
    labels = ['average pollock probability']
    # lines, labels = ax1.get_legend_handles_labels()
    proxy_artist = [plt.Line2D([0], [0], color='red', marker='o')]

    lines2, labels2 = ax2.get_legend_handles_labels()

    # Set the legend font size
    legend_fontsize = 9

    ax1.legend(proxy_artist + lines2 , labels + labels2 , loc='best',fontsize = legend_fontsize)

    # Set the title
    if save:
        plt.title(save)
        plt.savefig(save + '.png')
        
    # Display the plot
    plt.show()

def plot_year_vs_PMF_combined_scatter(full, save=False):
    df = full.copy()

    # Group the data by 'year' and calculate the means
    grouped_data = df.groupby('year')['pollock_prob'].mean().reset_index(name='mean')

    # Create a scatter plot
    fig, ax1 = plt.subplots()
    scat = ax1.scatter(df['year'], df['pollock_prob'], facecolor='none',edgecolors='blue')

    # Set x-axis labels as years
    ax1.set_xticks(grouped_data['year'])
    ax1.set_xticklabels(grouped_data['year'])

    # Set labels and title
    ax1.set_xlabel('Year')
    ax1.set_ylabel('PMF',color = 'blue')
    ax1.set_ylim([0.5, 1.05])

    # Calculate the mean of each year
    year_means = grouped_data['mean']

    # Plot the line connecting the means
    line = ax1.plot(grouped_data['year'], year_means, marker='x', color='blue')

    # Calculate the counts for each year
    year_counts = df['year'].value_counts().sort_index()

    # Create a twin axis for counts
    ax2 = ax1.twinx()
    ax2.plot(grouped_data['year'], year_counts.values, color='red', linestyle='--', label='Counts')
    ax2.set_ylabel('Counts',color = 'red')

    ax2.set_ylim(np.min(year_counts.values) - 1, int(np.max(year_counts.values)/np.min(year_means.values)+.8*np.max(year_counts.values)))

    labels = ['average PMF']
    # lines, labels = ax1.get_legend_handles_labels()
    # print(lines)
    proxy_artist = [plt.Line2D([0], [0], color='blue', marker='x')]

    lines2, labels2 = ax2.get_legend_handles_labels()
    # Set the legend font size
    legend_fontsize = 9
    proxy_artist3 = [mlines.Line2D([], [], color='none', marker='o', markersize=5, markerfacecolor='none', markeredgecolor='blue', label='Individual PMF')]
    labels3 = ['Individual PMF']
    ax1.legend(proxy_artist + lines2+proxy_artist3, labels + labels2+labels3, loc='best',fontsize = legend_fontsize)

    # ax1.legend(proxy_artist + lines2, labels + labels2, loc='best',fontsize = legend_fontsize)
    # Set the title
    if save:
        plt.title(save)
        plt.savefig(save + '.png')

    # Display the plot
    plt.show()

def remove_paintings(df, painting_names_to_remove):
    # Remove the rows with the specified painting names
    df = df.drop(painting_names_to_remove, level='painting')
    return df

def pollock_dial(test_img_prob, res_df,
               OneVote=False, save_name=None,legend=False,
               dial_markers = True,
               r_dial_markers_factor = 1.7,
               add_pointer = False,
               pointer_lw = 0.1,
               test_pointer = False,
               show_fig = True,
               dial_paintings = ['F34','D15', 'F102', 'A12', 'A66', 'P2(V)', 'P4(F)', 'C63', 'F65', 'JPCR_00173','P68(V)'],
               remove_special = ['A9(right)','JPCR_01088','P13(L)','P19(L)','P25(L)','P32(W)','P33(W Lowerqual)','P47(F Lowres)',
                                 'P65(L)','P75(V)','P77(J)','P80(J)','P86(S)','P105(J)','P106(V)','P115(F Lowres)','A14(Left)'],
               lines = False,
               round_to = 3,
               threshold = 0.56
               ):

    cat_dict = {'P':'P',
                'J':'P',
                'A':'A',
                'C':'A',
                'D':'A',
                'E':'A',
                'G':'A',
                'F':'A'}
    cat_df = make_cat_df(res_df, OneVote, cat_dict=cat_dict)
    paintings_in_cat_df = cat_df.index.get_level_values('painting').tolist()
    paintings_to_remove = list(set(remove_special).intersection(set(paintings_in_cat_df)))
    cat_df = remove_paintings(cat_df,paintings_to_remove)


    # pre-amble plot stuff
    fig_size_length = 12.7 #6.5
    max_height = 3.7  # max_height probably should have been called max radius = max distance above the colored circle
    start_height = 10  # probably should have been called start_radius
    total_height = start_height + max_height
    full_angle = 3*np.pi/2
    embed_length = 0.75 * start_height
    embed_area = 4 * start_height
    large_font_size = 30
    medium_font_size = 20
    break_start_factor = 1.45
    break_end_factor = 1.025 
    break_text_factor = 1.23 
    legend_position = (0.5, 0.1) 
    PMF_position = (0.5,0.15)
    dot_radii = start_height - 1.45*max_height
    pt_size = 24
    seed = 124
    aa_lines = 0.9
    lw_lines = 0.001


    start_angle = -convert_deg_to_rads(45)
    buffer_width_start = convert_deg_to_rads(1.25)
    buffer_width_end = convert_deg_to_rads(1.5)
    bound_angles = np.linspace(start_angle - buffer_width_start, full_angle + start_angle + buffer_width_end, 100)

    #describe data set
    pick_a_cat = ['P', 'A']#, 'N']
    categories = ['Pollock', 'Non-Pollock']# Poured', 'Non-Pollock Abstract']
    # print('categories are', pick_a_cat)
    colors = ['g', 'r', 'r', 'k']

    # find extreme performing comparators
    # (paintings with pollock_prob closest to 1 or 0)
    extreme_ones = find_comparators(cat_df, 1)
    extreme_zeros = find_comparators(cat_df, 0)
    extremes = extreme_ones + extreme_zeros


    # fig, ax = plt.subplots(1,1, figsize=(fig_size_length, fig_size_length),
    #                     constrained_layout=True,facecolor = 'white')
    fig, ax = plt.subplots(1,1, figsize=(fig_size_length, fig_size_length),
                        constrained_layout=False,facecolor = 'white')
    ax.axis('equal')
    ax.axis('off')

    ax = add_bound_circle(ax, radius=start_height,
                      angles=bound_angles, color='gray', ls='-')
    ax = add_arrow(ax, start_height - break_start_factor*max_height, start_angle - buffer_width_start)
    ax = add_arrow(ax, start_height - break_start_factor*max_height, full_angle + start_angle + buffer_width_end)
    ax = add_arrow(ax, start_height, start_angle - buffer_width_start,start_pos = start_height- break_end_factor*max_height)
    ax = add_arrow(ax, start_height, full_angle + start_angle + buffer_width_end,start_pos = start_height- break_end_factor*max_height)


    # add dial marker labels
    number_of_numbers = 5
    dial_dict = {'size': 3 * large_font_size/4}
    text_thetas = np.linspace(full_angle + start_angle, start_angle, number_of_numbers)
    dial_numbers_text = [f'{ii/100:0.2f}' for ii in np.linspace(0, 100, number_of_numbers)]
    rad_thetas = start_height + r_dial_markers_factor * max_height * np.ones(len(text_thetas))
    dial_numbers_x, dial_numbers_y = polar_to_cart(rad_thetas, text_thetas)
    for tex in np.arange(number_of_numbers):
        if dial_markers:
            ax.text(dial_numbers_x[tex], dial_numbers_y[tex], dial_numbers_text[tex], ha="center", va="center", fontdict=dial_dict)
        else:
            ax.text(dial_numbers_x[tex], dial_numbers_y[tex], ' ', ha="center", va="center", fontdict=dial_dict)
            dial_number_x, dial_number_y = polar_to_cart(start_height - break_text_factor*max_height, text_thetas)
            ax.text(dial_number_x[tex], dial_number_y[tex], dial_numbers_text[tex], ha="center", va="center", fontdict=dial_dict)
        if tex not in [0, number_of_numbers-1]:
            start_arr_dial = polar_to_cart(start_height - break_end_factor*max_height * len(pick_a_cat)/2, text_thetas[tex])
            # ax = add_arrow(ax, start_height, text_thetas[tex], start_pos=start_arr_dial, ls='dotted', color='gray')
            # start_arr_dial2 = polar_to_cart(start_height - 1.3*max_height * len(pick_a_cat)/2, text_thetas[tex])
            ax = add_arrow(ax, start_height - break_start_factor*max_height , text_thetas[tex], ls='dotted', color='gray')

    imp_xy = pd.DataFrame(index=extremes, columns=['x', 'y'])
    #make some extra buffer spacing down south to
    if not dial_markers:
        dial_number_x, dial_number_y = polar_to_cart(start_height + r_dial_markers_factor * max_height, 270)
        ax.text(0, -(start_height + r_dial_markers_factor * max_height), ' ', ha="center", va="center", fontdict=dial_dict)

    if dial_paintings:
        dial_paintings = dial_paintings[::-1]
        deg_start = -75
        deg_end = 255
        deg_window = int((deg_end-deg_start)/len(dial_paintings))
        painting_center_angles = range(int(deg_start + float(deg_window)/2),deg_end,deg_window)
        center_angle_dict = dict(zip(dial_paintings,painting_center_angles))
    for ii in np.arange(len(pick_a_cat)):
        bottom = start_height - max_height #* (ii+1)

        # possibly plot all slices for test image (add TEST to categories)
        if len(pick_a_cat[ii]) == 1:
            data = cat_df.loc[pick_a_cat[ii], 'pollock_prob']
        else:
            data = res.loc[res['painting_name'] == name, 'pollock_prob']

        thetas = data_to_angle(data, start_angle, full_angle)
        rads = place_radii_in_annulus(len(thetas), bottom + 0.05 * max_height,
                                        0.8 * max_height,seed = seed)
        x, y = polar_to_cart(rads, thetas)

        if dial_paintings:
            dial_painting_thetas = thetas[thetas.index.isin(dial_paintings)]
            dial_painting_rads = rads[thetas.index.isin(dial_paintings)]
            dial_x,dial_y = polar_to_cart(dial_painting_rads, dial_painting_thetas)
            dial_painting_names = dial_painting_thetas.index.tolist()
            if lines:
                for i,painting in enumerate(dial_painting_names):
                    ax = add_arrow(ax, start_height, convert_deg_to_rads(center_angle_dict[painting]),start_pos = (dial_x.iloc[i],dial_y.iloc[i]),
                                   ls=':',aa = aa_lines,lw = lw_lines)#,
                                #    head_width=0.05*dot_radii,head_length=0.05*dot_radii)
            c = ax.scatter(dial_x, dial_y, marker='o', facecolor=colors[ii], edgecolor=colors[ii], s=pt_size+25, zorder=10,label = '')


        selected_paintings = [ii for ii in extremes if ii in data.index]
        imp_xy.loc[selected_paintings, 'x'] = x.loc[selected_paintings]
        imp_xy.loc[selected_paintings, 'y'] = y.loc[selected_paintings]

        # scatter category and add bottom radius
        c = ax.scatter(x, y, c=colors[ii], alpha=0.5, s=pt_size, label=categories[ii])
        ax = add_bound_circle(ax, radius=bottom, angles=bound_angles,
                                color='black', ls='-', a=0.5)
        # ax = add_bound_circle(ax, radius=bottom, angles=bound_angles,
        #                         color=colors[ii], ls='--', a=0.5)


                       
        


    if legend:
        ax.legend(loc='center', fontsize=medium_font_size,bbox_to_anchor = legend_position)
    # display test
    text_x, text_y = polar_to_cart(start_height - break_text_factor*max_height, full_angle)
    font = {'fontname': 'Arial',
            'weight': 'bold',
            'size': large_font_size,
            }
    ax.text(text_x, text_y, 'PMF: ' + util.str_round(test_img_prob,round_to), fontdict=font, ha="center", va="center")
    # ax.text(PMF_position[0], PMF_position[1], f'PMF: {test_img_prob:.02f}', fontdict=font, ha="center", va="center")
    

    # add pointer to test figure
    data_angle = data_to_angle(test_img_prob, start_angle, full_angle)
    if test_pointer:
        pointer_line = start_height
    else:
        pointer_line = total_height

    
    if add_pointer:
        ax = add_arrow(ax, pointer_line, data_angle, color='k', lw=pointer_lw, zz=-1)
    else:
        x_test, y_test = polar_to_cart(dot_radii, data_angle)
        ax.scatter(x_test,y_test,c='black', alpha=1, s=30)
        ax = add_arrow(ax, 0.9*dot_radii, data_angle, color='k', zz=-1,ls= '-' ,head_width=0.07*dot_radii,head_length=0.07*dot_radii)

    #Add threshhold line
    thresh_angle = data_to_angle(threshold, start_angle, full_angle)
    ax = add_arrow(ax, start_height, thresh_angle, color='k', zz=-1,ls= (0, (5, 10)) )

    # save stuff
    if save_name is not None:
        if len(save_name.split('.')) == 1:
            save_name += '.png'
        plt.savefig(save_name,bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()

    return cat_df # extremes


import math

def get_radial_coordinates(radius, angle_degrees,offset = [256,256]):
    angle_radians = math.radians(angle_degrees)
    x = radius * math.cos(angle_radians) + offset[0]
    y = radius * math.sin(angle_radians) + offset[1]
    return x, y

# Example usage
def get_radial_pts(img,
                   deg_start = 0,
                   deg_end = 30,
                   deg_step= 1,
                   radius_bounds = [0.25,0.5],
                   offset = False #in pixels if not false
                   ):
    # Get the image dimensions
    height, width, _ = img.shape

    radii = []
    # radii.append((((width ** 2) + (height ** 2)) ** 0.5)/4)
    # radii.append((((width ** 2) + (height ** 2)) ** 0.5)*(2.5/4))
    if isinstance(radius_bounds[0],int) or isinstance(radius_bounds[0],np.int64):
        for r in radius_bounds:
            radii.append(r)
        # print('pixel',radii)
    else:
        for r in radius_bounds:
            radii.append(width*r)
        # print('proportion',radii)


    angle_degrees = range(deg_start,deg_end,deg_step)
    xs = []
    ys = []
    if not offset:
        offset=[width/2,height/2]

    for radius in radii:
        for angle in angle_degrees:
            
            x, y = get_radial_coordinates(radius, angle+180,offset=offset)
            xs.append(round(x))
            ys.append(round(y))
        angle_degrees = angle_degrees[::-1]
    return xs,ys


def get_px_center_of_dial(learner_results,
                        one_vote = True,
                        dial_markers=False,
                        r_dial_markers_factor =1.5,
                        dial_paintings = False,
                        lines = False,
                        temp_dir = 'Dial_Testing'
                        ):
    pollock_dial(0.5, learner_results,
            save_name=os.path.join(temp_dir,'width_dial.png'),
            OneVote = one_vote,
            legend = True,
            dial_markers=dial_markers,
            add_pointer = True,
            r_dial_markers_factor= r_dial_markers_factor,
            test_pointer=True,
            pointer_lw = 0.01,
            show_fig=False,
            dial_paintings =dial_paintings,
            lines=lines)
    width_dial = cv2.imread(os.path.join(temp_dir,'width_dial.png'))
    width_spectrogram = np.sum(np.sum(width_dial,axis=0),axis=1)
    x = np.argmin(width_spectrogram) #x center of dial
    pollock_dial(0.5, learner_results,
            save_name=os.path.join(temp_dir,'height_dial.png'),
            OneVote = one_vote,
            legend = True,
            dial_markers=dial_markers,
            add_pointer = True,
            r_dial_markers_factor= r_dial_markers_factor,
            test_pointer=True,
            pointer_lw = 10,
            show_fig = False,
            dial_paintings =dial_paintings,
            lines = lines
            )
    height_dial = cv2.imread(os.path.join(temp_dir,'height_dial.png'))
    height_spectrogram = np.sum(np.sum(height_dial,axis=1),axis=1)
    number = np.mean([np.max(height_spectrogram),np.min(height_spectrogram)])
    y= np.where(height_spectrogram < number)[0][-1] #y center of dial
    r= y -np.where(height_spectrogram < number)[0][0] #pixel radius of dial
    return x,y,int(r)

def put_image_on_image(background_image,foreground_image,coords):

    # Get the dimensions of the foreground image
    foreground_height, foreground_width, _ = foreground_image.shape

    # Calculate the top-left corner coordinates for placing the smaller image at the center
    top_left_x = coords[0] #- foreground_width // 2
    top_left_y = coords[1] #- foreground_height // 2

    # Place the smaller image onto the larger image
    background_image[top_left_y:top_left_y+foreground_height, top_left_x:top_left_x+foreground_width] = foreground_image
    return background_image
        
    
def make_dial_image(filename = 'Paintings/Processed/Raw/Max/A1_cropped_C0_R0.tif',
                   deg_start = 0,
                   deg_end = 30,
                   deg_step= 1,
                   radius_bounds = [0.25,0.5],
                   inverted = True,
                   offset = False,
                   resize_to = False #should be an image (like dial)
                   ):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    # # Concatenate the images horizontally
    # top_row = np.hstack((img, img))
    # bottom_row = np.hstack((img, img))

    # # Concatenate the top and bottom rows vertically
    # img = np.vstack((top_row, bottom_row))

    # Calculate the rotation matrix
    # height, width = img.shape[:2]
    # rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), deg_start/2, 1)

    # # Perform the rotation
    # img = cv2.warpAffine(img, rotation_matrix, (width, height))
    # if isinstance(resize_to,np.ndarray):
    #     img = cv2.resize(img, resize_to.shape[:2])
    if isinstance(img,type(None)):
        # masked_img = np.zeros_like(resize_to)
        if filename.split('_cropped')[0].endswith('white'):
            img = np.full((256, 256, 3), (255, 255, 255), dtype=np.uint8)
        else:
            img = np.full((256, 256, 3), (0, 0, 0), dtype=np.uint8)


    xs,ys = get_radial_pts(img,
                deg_start = deg_start,
                deg_end = deg_end,
                deg_step= deg_step,
                radius_bounds=radius_bounds,
                offset = offset
                )

    contour = np.array([[xii, yii] for xii, yii in zip(xs, ys)])
    #get the size that we want the image to be
    xy_diff = [np.max(contour[:,i])-np.min(contour[:,i]) for i in [0,1]]
    thumb_size = np.max(xy_diff)
    img = cv2.resize(img, (thumb_size,thumb_size))

    thumb_loc = [np.min(contour[:,0]),np.min(contour[:,1])]

    background = np.zeros_like(resize_to)
    full_img = put_image_on_image(background,img,thumb_loc)
    if inverted:
        mask    = np.zeros_like(resize_to)
        cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))
    else:
        mask    = np.ones_like(resize_to)*255
        cv2.fillPoly(mask, pts=[contour], color=(0,0,0))
    masked_img = cv2.bitwise_and(full_img, mask)
    return masked_img
    

def make_test_dial_image(filename = 'Paintings/Processed/Raw/Max/A1_cropped_C0_R0.tif',
                   radius = 0.5,
                   inverted = True,
                   offset = [0.5,0.5], #[width,height]
                   resize = False, #should be tuple (width,height) if not false
                   buffer_to = False, #Should be an img to buffer to if not False
                   inner_circle_factor = 0.35
                   ):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    height, width = img.shape[:2]
    radius = int(np.min([height,width])*radius)
    # Concatenate the images horizontally
    # top_row = np.hstack((img, img))
    # bottom_row = np.hstack((img, img))

    # # Concatenate the top and bottom rows vertically
    # img = np.vstack((top_row, bottom_row))
    offset = [offset[0]*width,offset[1]*height]



    xi,yi = get_radial_pts(img,
                deg_start = 0,
                deg_end = 360,
                deg_step= 1,
                radius_bounds=[int(radius)],
                offset = offset
                )
    contour = np.array([[xii, yii] for xii, yii in zip(xi, yi)])

    mask    = np.zeros_like(img)
    if inverted:
        mask    = np.zeros_like(img)
        cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))
    else:
        mask    = np.ones_like(img)*255
        cv2.fillPoly(mask, pts=[contour], color=(0,0,0))
    masked_img = cv2.bitwise_and(img, mask)

    if resize:
        masked_img = cv2.resize(masked_img, tuple([int(i*inner_circle_factor) for i in resize]))

    if isinstance(buffer_to,np.ndarray):
        background_image = np.zeros_like(buffer_to)
        # Calculate the position to place the foreground image at the center of the background image
        x = int((background_image.shape[1] - masked_img.shape[1]) / 2)
        y = int((background_image.shape[0] - masked_img.shape[0]) / 2)

        # Overlay the resized foreground image onto the background image at the calculated position
        background_image[y:y+masked_img.shape[0], x:x+masked_img.shape[1]] = masked_img


    return background_image
    
# not currently used but might be useful
def add_buffer_to_image(image,    
                        buffer_top = 10,
                        buffer_bottom = 20,
                        buffer_left = 30,
                        buffer_right = 40):
    # Calculate the new dimensions
    height, width = image.shape[:2]
    new_height = height + buffer_top + buffer_bottom
    new_width = width + buffer_left + buffer_right

    # Create a white canvas with the new dimensions
    canvas = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255

    # Calculate the positions to place the original image within the canvas
    x = buffer_left
    y = buffer_top

    # Place the original image onto the canvas
    canvas[y:y+height, x:x+width] = image
    return canvas

def make_dial_mask(img,
                   deg_start = -45,
                   deg_end = 225,
                   deg_step= 1,
                   radius_bounds = [0.25,0.5],
                   inverted = True,
                   offset = False,
                   inner_circle_pic = True,
                   inner_circle_factor = 0.35):    
    xs,ys = get_radial_pts(img,
                   deg_start = deg_start,
                   deg_end = deg_end,
                   deg_step= deg_step,
                   radius_bounds=radius_bounds,
                   offset = offset
                   )
    
    contour = np.array([[xii, yii] for xii, yii in zip(xs, ys)])
    if inverted:
        mask    = np.zeros_like(img)
        cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))
    else:
        mask    = np.ones_like(img)*255
        cv2.fillPoly(mask, pts=[contour], color=(0,0,0))

    masked_img = cv2.bitwise_and(img, mask)
    if inner_circle_pic:
        xi,yi = get_radial_pts(img,
                   deg_start = 0,
                   deg_end = 360,
                   deg_step= deg_step,
                   radius_bounds=[int(radius_bounds[0]*inner_circle_factor)],
                   offset = offset
                   )
        contour_inner = np.array([[xii, yii] for xii, yii in zip(xi, yi)])
        if inverted:
            mask2    = np.zeros_like(img)
            cv2.fillPoly(mask2, pts=[contour_inner], color=(255, 255, 255))
        else:
            mask2   = np.ones_like(img)*255
            cv2.fillPoly(mask2, pts=[contour_inner], color=(0,0,0))
        masked_img = cv2.bitwise_and(masked_img, mask2)
    
    return masked_img


def remove_white_space(image, threshold = 255,buffer = 5):

    array= [np.min(np.min(image,axis = i),axis = 1) for i in [0,1]]
    indices = [np.where(array[i] < threshold)[0] for i in [0,1]]

    # Get the first and last value
    first_values = [indices[i][0]-buffer for i in [0,1]]
    last_values = [indices[i][-1]+buffer for i in [0,1]]

    image = image[first_values[1]:last_values[1], first_values[0]:last_values[0]]



    return image



def make_pollock_dial(test_image_name,test_image_PMF,
                      test_image_path = 'Paintings/Processed/Raw/',
                      learner_folder = 'gcp_classifier-3_10-Max_color_10-15-2022',
                      test_ext = '_cropped_C0_R0.tif',
                      lines = False,
                      dial_paintings = ['F34','D15', 'C71', 'A12', 'A66', 'P2(V)', 'P4(F)', 'C63', 'F65', 'JPCR_00173','P68(V)'], # 'F102''P86(S)','F34', 'C2'
                      remove_special = ['A9(right)','JPCR_01088','P13(L)','P19(L)','P25(L)','P32(W)',
                      'P33(W Lowerqual)','P47(F Lowres)','P65(L)','P75(V)','P77(J)','P80(J)','P86(S)',
                      'P105(J)','P106(V)','P115(F Lowres)','A14(Left)'],
                      save_fig = True,
                      show_fig = True,
                      specify_results = False, #this can be a results df which would speed up the code slightly if running multiple times. Otehrwise grabs from directory
                      deg_start = -75,
                      deg_end = 255,
                      save_name = 'dial.png',
                      save_pre_path = '',
                      save_folder = os.path.join('Dial_Testing','figs'),
                      round_to = 3,
                      threshold = 0.56
                      ):
    
    temp_dir = 'Dial_Testing'
    r_dial_markers_factor = 2.5
    inner_circle_factor = 0.35
    # deg_start = -75
    # deg_end = 255
    one_vote = True
    deg_window = int((deg_end-deg_start)/len(dial_paintings))
    image_path = 'Paintings/Processed/Raw/Max/'
    ext = '_cropped_C0_R0.tif'
    test_image_folder = 'Max'
    test_image_file_path = os.path.join(test_image_path,test_image_folder,test_image_name + test_ext)
    paths = [os.path.join(image_path,file+ext) for file in dial_paintings]
    if isinstance(test_image_PMF,str):
        test_image_PMF = float(test_image_PMF)
    if isinstance(specify_results,pd.DataFrame):
        results = specify_results
    else:
        results = utils_fastai.get_learner_results(learner_folder = learner_folder,sets = ('valid','hold_out'))
    pollock_dial(test_image_PMF, results,
        save_name=os.path.join(temp_dir,'dial.png'),
        OneVote = one_vote,
        legend = True,
        dial_markers=False,
        add_pointer = False,
        r_dial_markers_factor= r_dial_markers_factor,
        test_pointer=False,
        pointer_lw = .1,
        dial_paintings = dial_paintings,
        remove_special = remove_special,
        lines = lines,
        show_fig=False,
        round_to = round_to,
        threshold = threshold
        )
    dial = cv2.imread(os.path.join(temp_dir,'dial.png'))
    # the function 'get_px_center_of_dial' wraps function 'pollock_dial' and is a common problem area
    x,y,r = get_px_center_of_dial(results,
                                  r_dial_markers_factor = r_dial_markers_factor,
                                  dial_paintings =dial_paintings,
                                  lines = lines
                                  ) 
    masked = make_dial_mask(dial,
                    deg_start = deg_start,
                    deg_end = deg_end,
                    deg_step= 1,
                    inner_circle_pic=True,
                    radius_bounds = [r,int(1.5*r)],
                    inverted = False,
                    offset = [x,y],
                    inner_circle_factor = inner_circle_factor
                    )
    center_img= make_test_dial_image(filename = test_image_file_path,
                   radius = 0.5,
                   inverted = True,
                   offset = (0.5,0.5),
                   resize = (2*r,2*r), #should be tuple (width,height) if not false
                   buffer_to = dial,
                   inner_circle_factor = inner_circle_factor
                   )
    imgs = []
    for i,file in enumerate(paths):
        imgs.append(make_dial_image(filename = file,
                        deg_start = i*deg_window+deg_start,
                        deg_end = (i+1)*deg_window+deg_start,
                        deg_step= 1,
                        radius_bounds = [r,1.5*r],
                        inverted = True,
                        offset = [x,y],
                        resize_to = dial
                        )
                    )
    result = np.sum(imgs, axis=0)
    combined = np.sum([result,masked,center_img], axis=0)
    finished= remove_white_space(combined)
    if save_name:
        if not os.path.exists(os.path.join(save_pre_path,save_folder)):
            os.makedirs(os.path.join(save_pre_path,save_folder))
        cv2.imwrite(os.path.join(save_pre_path,save_folder,save_name),finished)
    if show_fig:
        fig = plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(finished[:,:,[2,1,0]])
    #If you changed something in the pollock painting function and then things are misaligned check the function get_px_center_of_dial(), because I dont pass arguments through in a svelt way

def plot_res_vs_PMF(r,master):
    res_info = pd.merge(left = r[['painting','pollock_prob']], right = master[['file','px_per_cm_height','set','artist','remove_special']],left_on = 'painting',right_on = 'file')
    res = res_info[~res_info.remove_special & (res_info.set != 'train') & (res_info.artist != 'unknown')]
    fig = plt.figure(figsize = (10,6),facecolor='white')
    plt.scatter(res[res.artist == 'Pollock'].px_per_cm_height,res[res.artist == 'Pollock'].pollock_prob)
    plt.scatter(res[res.artist != 'Pollock'].px_per_cm_height,res[res.artist != 'Pollock'].pollock_prob)
    plt.xlabel('px_per_cm')
    plt.ylabel('PMF')
    plt.legend(['Pollock','non-Pollock'])
    plt.show()

def get_thresh_vs_MA(results,
                    master,
                    round_to = 2,
                    remove_special = ['P69(V)','P43(W)','JPCR_01030','P42(W)','JPCR_01031','A9(right)','JPCR_01088','P13(L)','P19(L)','P25(L)','P32(W)','P33(W Lowerqual)','P47(F Lowres)','P65(L)','P75(V)','P77(J)','P80(J)','P86(S)','P105(J)','P106(V)','P115(F Lowres)','A14(Left)'],
                    threshs = np.arange(0,1.01,0.01),
                    save = False,
                    ):
    MA = []
    for thresh in threshs:
        machine_accuracy_images = rubric.get_machine_accuracy(results,master,
                                                            remove_special = remove_special,
                                                            catagory = ('J','P'),
                                                            one_vote = True,
                                                            use_images = True,
                                                            percent = round_to,
                                                            binarize = False,
                                                            binarize_prob=0.5,
                                                            decision_prob=thresh)
        # machine_accuracy_images = str_round(machine_accuracy_images,round_to)
        MA.append(machine_accuracy_images)
    fig = plt.figure(figsize=(10,6),facecolor = 'white')
    plt.plot(threshs, MA)
    max_index = len(MA) - np.argmax(MA[::-1]) - 1
    # Add a vertical line at the position of the maximum value
    plt.axvline(x=threshs[max_index], color='r', linestyle='--')
    plt.xticks(np.append(np.arange(0,1.1,0.1),threshs[max_index]),rotation = 90)
    plt.xlabel('Clasification Threshold')
    plt.ylabel('Machine Accuracy (%)')
    if save:
        plt.savefig(fig, bbox_inches='tight')
        plt.close()
    else:
        plt.show()