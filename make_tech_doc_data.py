import stats
import utils_fastai
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize


def get_PMF_list(master,r,
                 save= 'PMF_List_T.csv',
                 master_catagories = ['file','title','year','catalog','set','remove_special'],
                #  list_columns = ['catalog','title','year','set','PMF'],
                 list_columns = ['catalog','title','year','set','PMF'],
                 extra_params = True):
    M = master[(master.is_pollock == 'True')].copy()
    PMF_List = pd.merge(left= M[master_catagories],right = r[['painting','pollock_prob']],left_on = 'file',right_on = 'painting')
    PMF_List = PMF_List[~PMF_List.remove_special]
    PMF_List.drop(columns=['file'], inplace=True)
    if extra_params:
        parameters = ['variation','mag','area_coverage','signal_uniformity']
        learner_folder = 'gcp_classifier-3_10-Max_color_10-15-2022'
        scales = pd.read_csv('runs/gcp_classifier-3_10-Max_color_10-15-2022/stats/scales.csv', index_col=0)
        full_results = utils_fastai.get_learner_results(learner_folder=learner_folder,sets = ('train','valid','hold_out')) 
        PJ = stats.get_stats(full_results,scales,M,categories = ('P','J'),verbose =True)
        PMF_List = pd.merge(left = PMF_List, right = PJ[['painting','variation','mag','area_coverage','signal_uniformity']],on = 'painting')
        for param in parameters:
            PMF_List[param] = PMF_List.groupby('catalog')[param].transform('mean')
            PMF_List[param] = PMF_List[param].round(2)
            # PMF_List['mag'] = PMF_List.groupby('catalog')['mag'].transform('mean')
            # PMF_List['area_coverage'] = PMF_List.groupby('catalog')['area_coverage'].transform('mean')
            # PMF_List['signal_uniformity'] = PMF_List.groupby('catalog')['signal_uniformity'].transform('mean')
        PMF_List = PMF_List.rename(columns={'variation': 'SI', 'mag': 'M', 'area_coverage': 'C', 'signal_uniformity': 'U'})
        list_columns = list_columns + ['SI','M','C','U']

    # 1. Calculate the mean of 'pollock_prob' grouped by 'catalog' and assign it to 'PMF'
    PMF_List['PMF'] = PMF_List.groupby('catalog')['pollock_prob'].transform('mean')
    # 2. Extract the year from the 'year' column
    PMF_List['year'] = pd.to_datetime(PMF_List['year']).dt.year
    # PMF_List.to_csv('PMF_intermediate_test.csv', index=False)
    # 3. Drop the 'pollock_prob' column
    PMF_List.drop(columns=['pollock_prob'], inplace=True)
    # 4. Remove duplicate rows based on 'catalog', 'title', 'year', and 'set'
    PMF_List.drop_duplicates(subset=['catalog', 'title', 'year', 'set'], keep='first', inplace=True)
    PMF_List['PMF'] = PMF_List['PMF'].round(2)
    # PMF_List = PMF_List[~PMF_List.remove_special][list_columns]
    PMF_List = PMF_List[list_columns]
    PMF_List.sort_values(by='year', ascending=True, inplace=True)
    if save:
        PMF_List.to_csv(save, index=False)
    return PMF_List

def make_tech_doc_data(round_to = 2,
                       save_path = 'tech_report'):
    results = utils_fastai.get_learner_results(sets = ('valid','hold_out')) 
    master = pd.read_parquet('master.parquet')
    stats.get_thesh_vs_MA(results,master,
                    round_to = round_to,
                    threshs = np.arange(0,1.01,0.01),
                    save = os.path.join(save_path,'thresh_vs_MA.png'),
                    )
    

def plot_stacked_images(image_paths, 
                        z_spacing = 0.1,
                        elevation=20, 
                        azimuth=-45,
                        vertical = True, 
                        lower_resolution=False,
                        show_grid = False,
                        save = False):
    # Create a 3D figure
    figsize = (10,10)
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Customize the view angle
    ax.view_init(elev=elevation, azim=azimuth)
    for i, item in enumerate(image_paths):
        # Load the image and normalize it to [0, 1] range
        image = mpimg.imread(item).astype(float) / 255.0

        if lower_resolution:
            # Resize the image to a lower resolution (e.g., 25% of original size)
            image = resize(image, (image.shape[0] // lower_resolution, image.shape[1] // lower_resolution), anti_aliasing=True)

        # Create a grid of points for the x-plane
        x = np.linspace(0, 1, image.shape[1])
        y = np.linspace(0, 1, image.shape[0])
        X, Y = np.meshgrid(x, y)

        # Plot the image on the x-plane using imshow
        if vertical:
            ax.plot_surface(X, Y, i*z_spacing*np.ones_like(X), rstride=1, cstride=1, facecolors=image, shade=False)
        else:
            ax.plot_surface(X, i*z_spacing*np.ones_like(X), -Y, rstride=1, cstride=1, facecolors=image, shade=False)

    if vertical:
        ax.set_box_aspect([1, 1, len(image_paths) * z_spacing], zoom=0.8)
    else:
        ax.set_box_aspect([1, len(image_paths) * z_spacing, 1], zoom=0.8)

    if not show_grid:
        # Remove grid lines
        ax.grid(False)

        # Remove axes
        ax.set_axis_off()

    if save:
        plt.savefig(save)

    # Show the 3D plot
    plt.show()