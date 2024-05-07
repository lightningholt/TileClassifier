from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import util
import os
from functools import reduce
import matplotlib.pyplot as plt
import utils_fastai
import matplotlib.patches as mpatches
import visualizer as vz
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from fractions import Fraction
import rubric
import seaborn as sns
from pathlib import Path


def get_scales(results,categories = ('P','J'),verbose = False,save = False,add_max = True):
    if categories:
        results = results[results.file.str.startswith(categories)].reset_index()
    results['slice_size'] = results['slice_size'].astype(str)
    hists = []
    paintings = sorted(list(set(results.painting_name)))
    for painting in paintings:
        util.print_if(verbose,painting)
        acc_by_group, painting_acc_at_sizes, error_group,error_painting = utils_fastai.painting_acc_by_slice_size(results,prob = True)
        slice_size_list = [str(item) for item in results.slice_size.tolist()]
        min_size = min([int(item) for item in slice_size_list if item.isnumeric()])
        painting_acc_index = [str(item) for item in painting_acc_at_sizes[painting].index]
        sizes = [int(item) for item in painting_acc_index if item.isdigit()]
        keys = [str(item) for item in range(min_size,np.max(sizes)+5,5)] + ['Max']
        data = {key:painting_acc_at_sizes[painting][key] for key in keys}
        hists.append(data)
    scales = pd.DataFrame(hists).transpose()
    scales.columns = paintings
    # Specify the key you want to be the last row
    specific_key = 'Max'

    # Extract the row corresponding to the specific key
    specific_row = scales.loc[scales.index == specific_key]

    # Remove the row from the DataFrame
    scales = scales.drop(index=specific_key)

    # Concatenate the extracted row with the DataFrame
    if add_max:
        scales = pd.concat([specific_row, scales])

    if save:
        scales.to_csv(save)
    return scales
from PIL import Image
def plot_scales(scales,
                groups = [('P','J'),('A','C','D','E','G')],
                labels = ['Pollocks','Non-Pollock Drips'],
                scatter = False, 
                width_factor = 1,
                show_fig = True,
                fontsize = 24,
                save = False,
                max = 275,
                colors = [(1, 0, 0), (0, 0, 0), (0, 0.5, 0)],
                selected_paths = False,
                slice_sizes = False,
                threshold = 0.56):

    cmap = vz.get_custom_colormap(colors =  colors)
    
    if max:
        scales = scales.loc[[str(i) for i in range(10,max,10)]]

    # Set up GridSpec for the figure
    fig = plt.figure(figsize = (14,10), facecolor='white')
    gs = fig.add_gridspec(3, 6, height_ratios=[0.5,0, 1], width_ratios=[0.027, 0.13, .25,.25,.25,.25])

    x = scales.index.tolist()
    y_values = []
    expression_groups = []
    colors = [cmap(255),cmap(0)]
    # y_errs = []
    width = 0.8
    for  group in groups:
        expression = '|^'.join(group)
        expression_group = scales.filter(regex = '^'+expression+'.*')
        expression_groups.append(expression_group)
        y_values.append(expression_group.mean(axis=1))
        # y_errs.append(expression_group.std(axis=1))
    ax = fig.add_subplot(gs[2, 2:], facecolor='white')
    for i, y in enumerate(y_values):
        # y_err = y_errs[i]  # Calculate the standard deviation as the error
        # plt.plot(x, y,  label=labels[i])
        ax.bar(x, y,  label=labels[i],width = width,color = cmap(y))

        arrow_y_value = np.mean(y)

        # Get the positions of the x-labels
        x_positions = range(len(x))
        # arrowprops = dict(arrowstyle='->',linewidth=2.5,color = colors[i])

        # ax.annotate('PMF', xy=(len(x), arrow_y_value),xytext =(len(x)+1,arrow_y_value),fontsize=16,
        #                     arrowprops = arrowprops)
        # ax.axhline(y=np.mean(y), linestyle='dotted', color=colors[i])

        # if scatter: # doesn't quite work yet but ideas for how to.
        #     massaged_data = pd.melt(expression_groups[i].transpose().reset_index().drop('index',axis=1))
        #     plt.scatter(,massaged_data.value)
            # massaged_data.plot(kind='scatter', x='variable', y='value',figsize= (14,10))
        #     plt.scatter(x,expression_groups[i].values)

        width *= width_factor

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
    
    cax = fig.add_subplot(gs[2, 0])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap),cax=cax)
    custom_ticks = [0,threshold,1]  # Replace with your desired custom tick positions
    cbar.set_ticks(custom_ticks)
    

    cbar.ax.yaxis.set_tick_params(labelsize=16)

    cbar.ax.yaxis.set_ticks_position('left')


    plt.subplots_adjust(bottom=0.3, left=0.2, hspace=0, wspace=0) 

    # plt.legend(fontsize = fontsize-10)
    ax.set_ylim(0,1)
    ax.set_xticklabels([str(i) for i in range(10,275,10)],fontsize=fontsize-10,rotation = 90)
    ax.set_yticklabels([str(round(i,1)) for i in np.linspace(0,1,6)],fontsize=fontsize-10)
    # plt.xticks(['Max'] + [str(i) for i in range(15,360,10)],fontsize=fontsize-10)
    # ax.set_xticks([str(i) for i in range(10,275,10)],fontsize=fontsize-10)
    ax.set_xlim(-1,len(x))
    # plt.xlim(0,270)
    ax.set_xlabel('Tile Size[cm]',fontsize = fontsize)
    ax.set_ylabel('Pollock Signature',fontsize = fontsize)
    if save:
        plt.savefig(save,dpi=300, bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close()

    return y_values
# def plot_scales(scales,
#                 groups = [('P','J'),('A','C','D','E','G')],
#                 labels = ['Pollocks','Non-Pollock Drips'],
#                 scatter = False, 
#                 width_factor = 1,
#                 show_fig = True,
#                 fontsize = 24,
#                 save = False,
#                 max = 275,
#                 colors = [(1, 0, 0), (0, 0, 0), (0, 0.5, 0)]):
#     if not colors: # alternatively colors = ['blue','red']
#         colormap = plt.get_cmap('tab10')
#         num_colors = 10
#         colors = [colormap(i) for i in range(num_colors)]
#     else:
#         cmap = vz.get_custom_colormap(colors =  colors)
    
#     if max:
#         scales = scales.loc[[str(i) for i in range(10,max,10)]]

#     fig, ax = plt.subplots(figsize = (14,6),facecolor = 'white')
#     x = scales.index.tolist()
#     y_values = []
#     expression_groups = []
#     colors = [cmap(255),cmap(0)]
#     # y_errs = []
#     width = 0.8
#     for  group in groups:
#         expression = '|^'.join(group)
#         expression_group = scales.filter(regex = '^'+expression+'.*')
#         expression_groups.append(expression_group)
#         y_values.append(expression_group.mean(axis=1))
#         # y_errs.append(expression_group.std(axis=1))
#     for i, y in enumerate(y_values):
#         # y_err = y_errs[i]  # Calculate the standard deviation as the error
#         # plt.plot(x, y,  label=labels[i])
#         ax.bar(x, y,  label=labels[i],width = width,color = cmap(y))

#         arrow_y_value = np.mean(y)

#         # Get the positions of the x-labels
#         x_positions = range(len(x))
#         arrowprops = dict(arrowstyle='->',linewidth=2.5,color = colors[i])

#         ax.annotate('', xy=(len(x), arrow_y_value),xytext =(len(x)+1,arrow_y_value),
#                             arrowprops = arrowprops)
#         # plt.axhline(y=np.mean(y), linestyle='dotted', color=colors[i])
#         # if scatter: # doesn't quite work yet but ideas for how to.
#         #     massaged_data = pd.melt(expression_groups[i].transpose().reset_index().drop('index',axis=1))
#         #     plt.scatter(,massaged_data.value)
#             # massaged_data.plot(kind='scatter', x='variable', y='value',figsize= (14,10))
#         #     plt.scatter(x,expression_groups[i].values)
#         width *= width_factor
#     cax = fig.add_axes([0.04, 0.12, 0.015, 0.76])
#     cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap),cax=cax)
#     custom_ticks = [0,0.56,1]  # Replace with your desired custom tick positions
#     cbar.set_ticks(custom_ticks)
#     cbar.ax.yaxis.set_ticks_position('left')




#     # plt.legend(fontsize = fontsize-10)
#     ax.set_ylim(0,1)
#     ax.set_xticklabels([str(i) for i in range(10,275,10)],fontsize=fontsize-10,rotation = 90)
#     ax.set_yticklabels([str(round(i,1)) for i in np.linspace(0,1,6)],fontsize=fontsize-10)
#     # plt.xticks(['Max'] + [str(i) for i in range(15,360,10)],fontsize=fontsize-10)
#     # ax.set_xticks([str(i) for i in range(10,275,10)],fontsize=fontsize-10)
#     ax.set_xlim(-1,len(x))
#     # plt.xlim(0,270)
#     ax.set_xlabel('Tile Size (cm)',fontsize = fontsize)
#     ax.set_ylabel('Pollock Signature',fontsize = fontsize)
#     if save:
#         plt.savefig(save,dpi=300, bbox_inches="tight")
#     if show_fig:
#         plt.show()
#     else:
#         plt.close()

def get_rmse_values(df,rescale = False):
    if isinstance(df,(list,pd.core.series.Series)):
        rmse_values = calculate_rmse(df, np.full(len(df),np.mean(df)))
    else:
        rmse_values = np.sqrt(df.apply(lambda x: mean_squared_error(x.dropna(), np.full(len(x.dropna()), x.mean())), axis=0))
    if rescale:
        rmse_values = 1 - (2 * (rmse_values - 0))
    if isinstance(df,(list,pd.core.series.Series)):
        return rmse_values
    else:
        df = pd.DataFrame({'painting': list(rmse_values.index),'variation' : list(rmse_values)})  
        return df

def calculate_rmse(true_values, predicted_values):
    n = len(true_values)
    squared_errors = [(true_values[i] - predicted_values[i])**2 for i in range(n)]
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse

def get_signal_uniformity(results,viz_path_start = 'viz/PJ',viz_path_end = 'overlap_resize',verbose = False,rescale = True):
    painting_list = sorted(list(set(results.painting_name)))
    SU = []
    for painting in painting_list:
        util.print_if(verbose,painting)
        viz_path = os.path.join(viz_path_start,painting+'_' +viz_path_end)
        cr = np.load(os.path.join(viz_path,'combined.npy'))
        y = [np.nan,np.nan]
        for i in range(2):
            y[i] = np.nanmean(cr,axis = i)
            y[i] = np.sqrt(mean_squared_error(y[i],np.full(len(y[i]), np.mean(y[i]))))
            if rescale:
                y[i] = 1 - (2 * (y[i] - 0))
        SU.append(np.mean(y))
    return pd.DataFrame({'painting': painting_list,
                        'signal_uniformity' : SU})

def get_mags(df):
    # Perform the calculations for each column
    result = {}
    for col in df.columns:
        numerator = float(df.iloc[df[col].count() - 1].name)
        denominator = 10.0
        division = numerator / denominator
        result[col] = division
    mags = pd.DataFrame({'painting' : list(result.keys()),
                                            'mag': list(result.values())})
    return mags

def get_tile_coverage(df,threshold=0.56):
    # Calculate the percentage of rows above the threshold value
    cov = df.groupby('painting_name')['pollock_prob'].apply(lambda x: (x > threshold).mean())
    coverage = pd.DataFrame({'painting':list(cov.keys()),
                             'pollock_coverage':list(cov)})
    return coverage

def get_area_coverage(results,viz_path_start = 'viz/PJ',viz_path_end = 'overlap_resize',threshold=0.56,verbose = False):
    painting_list = sorted(list(set(results.painting)))
    area_coverages = []
    for painting in painting_list:
        util.print_if(verbose,painting)
        viz_path = os.path.join(viz_path_start,painting+'_' +viz_path_end)
        cr = np.load(os.path.join(viz_path,'combined.npy'))
        area_coverages.append(cr[cr >= threshold].size/cr.size)
    return pd.DataFrame({'painting': painting_list,
                        'area_coverage' : area_coverages})

def filter_master(df,column = 'file',strings_to_exclude = ['Left','left','right','Right']):
    #typical input is master, but works for other dfs too

    # Create the regular expression pattern
    pattern = '|'.join(strings_to_exclude)

    # Filter out rows with painting names containing any of the strings
    filtered_df = df[~df[column].str.contains(pattern)]

    # Print the filtered DataFrame
    return filtered_df

def get_aspect(master,results,filter = False):
    if filter:
        M = filter_master(master,column = 'file',strings_to_exclude = ['Left','left','right','Right'])
    else:
        M = master
    aspect = pd.DataFrame({'painting': list(M[M.file.isin(results.painting_name)].file),
              'aspect' : list(M[M.file.isin(results.painting_name)].height_cm/M[M.file.isin(results.painting_name)].width_cm)})
    return aspect

def get_canvas_size(master,results,filter = False):
    if filter:
        M = filter_master(master,column = 'file',strings_to_exclude = ['Left','left','right','Right'])
    else:
        M = master
    canvas_size = pd.DataFrame({'painting': list(M[M.file.isin(results.painting_name)].file),
              'canvas_size_m2' : list(M[M.file.isin(results.painting_name)].height_cm/100*M[M.file.isin(results.painting_name)].width_cm/100)})
    return canvas_size


def get_stats(results,scales,master,categories = ('P','J'),verbose = False,save= False,viz_path_start = 'viz/PJ',viz_path_end  = 'overlap_resize',skip_viz= False,threshold = 0.56,do_year = True):
    #Pollock timeline data
    #get values for each painting, then we'll merge and split based on date
    #If you want to just get the PJ's (necessary for area coverage until we run everything)
    if categories:
        results = results[results.file.str.startswith(categories)].reset_index()

    ##get SIGNATURE
    #get PMF
    util.print_if(verbose,'PMF')
    classification_thresh = threshold
    binarize_prob = 0.5
    one_vote = True
    PMF = util.vote_system(results, one_vote=one_vote, binarize=False, binarize_prob=binarize_prob, decision_prob=classification_thresh)[['painting','pollock_prob']].rename(columns = {'pollock_prob':'PMF'})

    ## get SCALING
    #get Variation
    util.print_if(verbose,'variation')
    variation = get_rmse_values(scales,rescale =True)

    #get magnification
    util.print_if(verbose,'magnification')
    mags = get_mags(scales)


    #get coverage
    if not skip_viz:
        util.print_if(verbose,'coverage')
        # pollock_coverage = get_tile_coverage(results,threshold=0.56)
        pollock_coverage =  get_area_coverage(results,viz_path_start = viz_path_start,viz_path_end = viz_path_end,threshold=threshold,verbose=False)

    #get signal uniformity
    if not skip_viz:
        util.print_if(verbose,'signal uniformity')
        signal_uniformity = get_signal_uniformity(results,viz_path_start = viz_path_start,viz_path_end = viz_path_end,verbose = False,rescale = True)

    ##get COMPOSITION
    #get aspect ratio  #doesn't automatically filter away left/right paintings. calculates based on dimensions given
    util.print_if(verbose,'aspect')
    aspect = get_aspect(master,results,filter = False)

    #get canvas size #doesn't automatically filter away left/right paintings. calculates based on dimensions given
    util.print_if(verbose,'canvas size')
    canvas_size = get_canvas_size(master,results,filter = False)

    ##get Productivity
    #get number
    util.print_if(verbose,'number')

    util.print_if(verbose,'compile')
    if skip_viz:
        dfList = [PMF,variation,mags,aspect,canvas_size]
    else:
        dfList = [PMF,variation,mags,pollock_coverage,signal_uniformity,aspect,canvas_size]

    stats = reduce(lambda x, y: pd.merge(x, y, on = 'painting'), dfList)
    stats = pd.merge(stats, master[['file','year']], how="inner", left_on='painting', right_on='file')
    if do_year:
        stats['year'] = stats['year'].dt.year.astype(float).astype('Int64')
    if save:
        stats.to_csv(save)
    return stats

def plot_stats(stats,
               border_line_color = 'black',
               inter_line_color = 'black',
               border_line_width = 4,
               inter_line_width = 2,
               col_names = ['PMF','SI','M','C','U','AR','A','N'],
               subplot_type = ['solo','spacing','top','bottom','spacing','top','bottom','spacing','top','bottom','spacing','solo'],
               plot_line_color = '#c1272d',
               scatter_color = 'black',#'#008176',
               fontsize = 18,
               ticklabel_fontsize = 13,
               corr_fontsize = 18,
               group_fontsize = 16,
               plotlabel_fontsize = 15,
               correlation=True,
               groups = ['Matching','Scaling','Spatial','Composition','Productivity'],
               corr_x = .92,
               corr_y = 1.15,
               corr_bbox_props = dict(boxstyle='circle', fc='none', ec='black', linewidth=2),
               corr_label = False,#'Corr',
               groups_x = 1.01,
               groups_rot = 270,
               figsize=(12, 15),
               region_alpha = 0.2,
               region_colors = ['red','blue','green','yellow'],
               regions = [(1942, 1946.5), (1946.5, 1949.5), (1949.5, 1952.5), (1952.5, 1955.5)],
               test_color = 'black',
               test_width = 1,
               test_linestyle = ':',
               plot_label = ['a','b','c','d','e','f','g','h'],
               plot_label_x = 0.975,
               label_bbox_props = dict(boxstyle='square', fc='none', ec='black', linewidth=1),
               vspace = .1,
               ylims = [[-.1,1.1],[-.1,1.1],[-.1,27],[-.1,1.1],[-.1,1.1],[-.1,3.5],[-1,17],[-.1,80]],
               yticks = [[0,1],[0,1],[0,10,20],[0,1],[0,1],[0,1,2,3],[0,5,10,15],[0,25,50,75]],
               ylabel_x = -0.07,
               arrowprops = dict(arrowstyle='->',linewidth=2.5),
               arrow_length = 0.75,
               test_stats_or_list = [0.56, 0.33, 5.0, 0.84, 0.90, 1.23, 0.35, 1943],   #PJ[PJ.painting == 'P2(V)']    
               show_fig = True,
               save = False,
               corr_type = 'pearson'
               ):
    if isinstance(test_stats_or_list,pd.DataFrame):
        columns = ['PMF', 'variation', 'mag','area_coverage', 'signal_uniformity',  'aspect', 'canvas_size_m2','year']
        test_stats_or_list = util.get_row_values(test_stats_or_list, columns)
    if correlation:
        # corr = stats.corr(numeric_only=True).PMF.tolist()
        corr = stats.corr(method = corr_type).PMF.tolist()
    columns = list(stats.columns[1:-2])
    if not col_names:
        col_names = columns + ['number of paintings']
    # ylims = [(0.56,1.1),(.75,1.1),(-0.01,0.1),(3,17),(0.65,2.1),(3000,64000)]

    gridspec = dict(hspace=0.0, height_ratios=[1,vspace, 1, 1, vspace, 1, 1, vspace, 1, 1, vspace, 1])
    fig, axs = plt.subplots(nrows=len(columns)+1+4, ncols=1,figsize=figsize,facecolor = 'white',gridspec_kw=gridspec)

    # Adjust the border thickness and color for each axis
    for i, ax in enumerate(axs):
        ax.spines['left'].set_linewidth(border_line_width)
        ax.spines['right'].set_linewidth(border_line_width)
        ax.spines['left'].set_color(border_line_color)
        ax.spines['right'].set_color(border_line_color)

        if subplot_type[i] == 'top':
            ax.spines['top'].set_color(border_line_color)
            ax.spines['bottom'].set_color(inter_line_color)
            ax.spines['bottom'].set_linewidth(inter_line_width)
            ax.spines['top'].set_linewidth(border_line_width)
        elif subplot_type[i] == 'bottom':
            ax.spines['top'].set_color(inter_line_color)
            ax.spines['bottom'].set_color(border_line_color)
            ax.spines['bottom'].set_linewidth(border_line_width)
            ax.spines['top'].set_linewidth(inter_line_width)
        elif subplot_type[i] == 'solo':
            ax.spines['top'].set_color(border_line_color)
            ax.spines['bottom'].set_color(border_line_color)
            ax.spines['bottom'].set_linewidth(border_line_width)
            ax.spines['top'].set_linewidth(border_line_width)
        elif subplot_type[i] == 'mid':
            ax.spines['top'].set_color(inter_line_color)
            ax.spines['bottom'].set_color(inter_line_color)
            ax.spines['bottom'].set_linewidth(inter_line_width)
            ax.spines['top'].set_linewidth(inter_line_width)
        elif subplot_type[i] == 'spacing':
            ax.set_visible(False)
        else:
            print('unreconized subplot type')

    i = 0
    for j,ax in enumerate(axs[:-1]):
        # print(i,j)
        # print(subplot_type[j])
        if j not in [1,4,7,10]:
            ax.plot(stats.groupby('year')[columns[i]].mean(),c=plot_line_color)
            ax.scatter(stats['year'], stats[columns[i]], facecolor=scatter_color,edgecolors=scatter_color)
            # ax.axhline(y=test_stats_or_list[i],color = test_color,linestyle=test_linestyle,linewidth = test_width)
            # ax.annotate('', xy=(min(regions[0]), test_stats_or_list[i]),xytext =(min(regions[0])+arrow_length,test_stats_or_list[i]),
            #             arrowprops = arrowprops)
            # ax.set_ylabel(col_names[i],fontsize = fontsize)
            ax.annotate(col_names[i],xy = (ylabel_x, 0.5),fontsize = fontsize,xycoords='axes fraction',horizontalalignment='left', verticalalignment='center',rotation=90)
            ax.set_xlim(min(regions[0]),max(regions[-1]))
            ax.set_xticks([])
            ax.set_ylim(ylims[i])
            ax.set_yticks(yticks[i])
            ax.tick_params(axis='both', labelsize=ticklabel_fontsize)
            # Add text label next to each subplot
            if plot_label:
                ax.annotate(plot_label[i], xy=(plot_label_x, 0.965), xycoords='axes fraction', fontsize=plotlabel_fontsize,
                            horizontalalignment='left', verticalalignment='top')
                ax.annotate('  ', xy=(plot_label_x, 0.965), xycoords='axes fraction', fontsize=plotlabel_fontsize,
                            horizontalalignment='left', verticalalignment='top',bbox=label_bbox_props)
            if correlation:
                # circle = mpatches.Circle((corr_x, 0.5), radius=0.1, edgecolor='black', facecolor='none',transform=ax.transAxes)
                # ax.add_patch(circle)
                if i ==0 and corr_label:
                    ax.annotate(corr_label, xy=(corr_x, corr_y), xycoords='axes fraction', fontsize=corr_fontsize,
                        horizontalalignment='left', verticalalignment='center',weight='bold',bbox=corr_bbox_props)
                if i != 0:
                    ax.annotate(util.round_special_no_zero_neg(corr[i],1), xy=(corr_x, 0.5), xycoords='axes fraction', fontsize=corr_fontsize,
                            horizontalalignment='left', verticalalignment='center',weight='bold',bbox=corr_bbox_props)
            if groups:
                if i==0:
                    ax.annotate(groups[0], xy=(groups_x, 0.5), xycoords='axes fraction', fontsize=group_fontsize,
                        horizontalalignment='left', verticalalignment='center',rotation = groups_rot)
                if i==1:
                    ax.annotate(groups[1], xy=(groups_x, 0), xycoords='axes fraction', fontsize=group_fontsize,
                        horizontalalignment='left', verticalalignment='center',rotation = groups_rot)
                if i==3:
                    ax.annotate(groups[2], xy=(groups_x, 0), xycoords='axes fraction', fontsize=group_fontsize,
                        horizontalalignment='left', verticalalignment='center',rotation = groups_rot)
                if i==5:
                    ax.annotate(groups[3], xy=(groups_x, 0), xycoords='axes fraction', fontsize=group_fontsize,
                        horizontalalignment='left', verticalalignment='center',rotation = groups_rot)
            for region,region_color in zip(regions,region_colors):
                ax.axvspan(region[0], region[1], facecolor=region_color, alpha=region_alpha)
            i += 1



    counts = stats.year.value_counts().sort_index()
    axs[-1].plot(list(counts.keys()), list(counts.values),c = plot_line_color)

    axs[-1].set_xlabel('Year',fontsize = fontsize)
    # axs[-1].set_ylabel(col_names[-1],fontsize = fontsize)
    axs[-1].annotate(col_names[-1],xy = (-0.08, 0.5),fontsize = fontsize,xycoords='axes fraction',horizontalalignment='left', 
                     verticalalignment='center',rotation=90)
    axs[-1].set_xticks([1943+i for i in range(12)])
    axs[-1].set_xticklabels([str(1943+i) for i in range(12)])
    axs[-1].tick_params(axis='both', labelsize=ticklabel_fontsize)
    if plot_label:
        axs[-1].annotate(plot_label[-1], xy=(plot_label_x, 0.965), xycoords='axes fraction', fontsize=plotlabel_fontsize,
                    horizontalalignment='left', verticalalignment='top')
        axs[-1].annotate('  ', xy=(plot_label_x, 0.965), xycoords='axes fraction', fontsize=plotlabel_fontsize,
                    horizontalalignment='left', verticalalignment='top',bbox=label_bbox_props)
    if correlation:
        axs[-1].annotate(util.round_special_no_zero_neg(corr[-1],1), xy=(corr_x, 0.5), xycoords='axes fraction', fontsize=corr_fontsize,
                    horizontalalignment='left', verticalalignment='center',weight='bold',bbox=corr_bbox_props)
    if groups:
        axs[-1].annotate(groups[-1], xy=(groups_x, 0.5), xycoords='axes fraction', fontsize=group_fontsize,
                    horizontalalignment='left', verticalalignment='center',rotation = groups_rot)      
    for region,region_color in zip(regions,region_colors):
            axs[-1].axvspan(region[0], region[1], facecolor=region_color, alpha=region_alpha)  

    axs[-1].set_yticks(yticks[-1])
    axs[-1].set_xlim(min(regions[0]),max(regions[-1]))
    plt.subplots_adjust(hspace=0.0)
    if save:
        plt.savefig(save,dpi=300,bbox_inches = 'tight',pad_inches = 0.1)
        # plt.savefig(save,dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_PMFvsStats(stats,
               border_line_color = 'black',
               inter_line_color = 'black',
               border_line_width = 4,
               inter_line_width = 2,
               col_names = ['SI','M','C','U'],
               subplot_type = ['top','bottom','spacing','top','bottom'],
               plot_line_color = '#c1272d',
               scatter_color = 'black',#'#008176',
               fontsize = 18,
               ticklabel_fontsize = 13,
               corr_fontsize = 18,
               group_fontsize = 16,
               plotlabel_fontsize = 15,
               correlation=False,
               groups = ['Scaling','Spatial'],
               corr_x = .92,
               corr_y = 1.15,
               corr_bbox_props = dict(boxstyle='circle', fc='none', ec='black', linewidth=2),
               corr_label = False,#'Corr',
               groups_x = 1.01,
               groups_rot = 270,
               figsize=(12, 15),
               region_alpha = 0.5,
            #    region_colors = ['red','blue','yellow','green'],
            #    regions = [(0.5, .557), (.557, .608), (.608, .895), (.895, 1.06)],
            #    region_colors = [(0.2, 0.6, 1.0),(0.2, 0.8, 0.2),(1.0, 0.6, 0.2),(0.6, 0.2, 0.8)],
               region_colors = [(0.8, 0.2, 0.2), (0.2, 0.2, 0.6), (1.0, 0.6, 0.2), (0.2, 0.6, 0.2)],
               regions = [(0.5, .557), (.557, .608), (.608, .895), (.895, 1.06)],
            #    region_colors = ['red',(0,1,0)],
            #    regions = [(0.5, .557), (.557, 1.06)],
            #    region_colors = ['white','white'],
            #    regions = [(0.53, .557), (.557, 1.06)],
               test_color = 'black',
               test_width = 2,
               test_linestyle = ':',
               plot_label = ['a','b','c','d'],
               plot_label_x = 0.013,
               label_bbox_props = dict(boxstyle='square', fc='none', ec='black', linewidth=1),
               vspace = .1,
               ylims = [[-.1,1.1],[-.1,27],[-.1,1.1],[-.1,1.1]],
               yticks = [[0,1],[0,10,20],[0,1],[0,1]],
               ylabel_x = -0.07,
               arrowprops = dict(arrowstyle='->',linewidth=2.5),
               arrow_length = 0.05,
               test_stats_or_list = [0.56, 0.33, 5.0, 0.84],   #PJ[PJ.painting == 'P2(V)']    
               show_fig = True,
               save = False,
               corr_type = 'pearson'
               ):
    columns = ['variation', 'mag','area_coverage', 'signal_uniformity']
    if isinstance(test_stats_or_list,pd.DataFrame):
        # columns = ['PMF', 'variation', 'mag','area_coverage', 'signal_uniformity',  'aspect', 'canvas_size_m2','year']
        test_stats_or_list = util.get_row_values(test_stats_or_list, columns)
    if correlation:
        # corr = stats.corr(numeric_only=True).PMF.tolist()
        corr = stats.corr(method=corr_type)[['PMF']+columns].PMF[1:].tolist()
    # columns = list(stats.columns[1:-2])
    if not col_names:
        col_names = columns + ['number of paintings']
    # ylims = [(0.56,1.1),(.75,1.1),(-0.01,0.1),(3,17),(0.65,2.1),(3000,64000)]

    gridspec = dict(hspace=0.0, height_ratios=[1, 1, vspace, 1, 1])
    fig, axs = plt.subplots(nrows=len(columns)+1, ncols=1,figsize=figsize,facecolor = 'white',gridspec_kw=gridspec)

    # Adjust the border thickness and color for each axis
    for i, ax in enumerate(axs):
        ax.spines['left'].set_linewidth(border_line_width)
        ax.spines['right'].set_linewidth(border_line_width)
        ax.spines['left'].set_color(border_line_color)
        ax.spines['right'].set_color(border_line_color)

        if subplot_type[i] == 'top':
            ax.spines['top'].set_color(border_line_color)
            ax.spines['bottom'].set_color(inter_line_color)
            ax.spines['bottom'].set_linewidth(inter_line_width)
            ax.spines['top'].set_linewidth(border_line_width)
        elif subplot_type[i] == 'bottom':
            ax.spines['top'].set_color(inter_line_color)
            ax.spines['bottom'].set_color(border_line_color)
            ax.spines['bottom'].set_linewidth(border_line_width)
            ax.spines['top'].set_linewidth(inter_line_width)
        elif subplot_type[i] == 'solo':
            ax.spines['top'].set_color(border_line_color)
            ax.spines['bottom'].set_color(border_line_color)
            ax.spines['bottom'].set_linewidth(border_line_width)
            ax.spines['top'].set_linewidth(border_line_width)
        elif subplot_type[i] == 'mid':
            ax.spines['top'].set_color(inter_line_color)
            ax.spines['bottom'].set_color(inter_line_color)
            ax.spines['bottom'].set_linewidth(inter_line_width)
            ax.spines['top'].set_linewidth(inter_line_width)
        elif subplot_type[i] == 'spacing':
            ax.set_visible(False)
        else:
            print('unreconized subplot type')

    i = 0
    for j,ax in enumerate(axs):
        # print(i,j)
        # print(subplot_type[j])
        if j not in [2]:
            for region,region_color in zip(regions,region_colors):
                ax.axvspan(region[0], region[1], facecolor=region_color, alpha=region_alpha,zorder=0)
            # ax.plot(stats.groupby('PMF')[columns[i]].mean(),c=plot_line_color)
            ax.scatter(stats['PMF'], stats[columns[i]], facecolor=scatter_color,edgecolors=scatter_color,zorder=1)
            ax.axhline(y=np.mean(stats[columns[i]]),xmin = 0.045,color = test_color,linestyle=test_linestyle,linewidth = test_width,zorder=1)
            # ax.annotate('', xy=(min(regions[0]), test_stats_or_list[i]),xytext =(min(regions[0])+arrow_length,test_stats_or_list[i]),
            #             arrowprops = arrowprops)
            # ax.set_ylabel(col_names[i],fontsize = fontsize)
            ax.annotate(col_names[i],xy = (ylabel_x, 0.5),fontsize = fontsize,xycoords='axes fraction',horizontalalignment='left', verticalalignment='center',rotation=90)
            ax.set_xlim(min(regions[0]),max(regions[-1]))
            ax.set_xticks([])
            ax.set_ylim(ylims[i])
            ax.set_yticks(yticks[i])
            ax.tick_params(axis='both', labelsize=ticklabel_fontsize)
            # Add text label next to each subplot
            if plot_label:
                ax.annotate(plot_label[i], xy=(plot_label_x, 0.965), xycoords='axes fraction', fontsize=plotlabel_fontsize,
                            horizontalalignment='left', verticalalignment='top')
                ax.annotate('  ', xy=(plot_label_x, 0.965), xycoords='axes fraction', fontsize=plotlabel_fontsize,
                            horizontalalignment='left', verticalalignment='top',bbox=label_bbox_props)
            if correlation:
                # circle = mpatches.Circle((corr_x, 0.5), radius=0.1, edgecolor='black', facecolor='none',transform=ax.transAxes)
                # ax.add_patch(circle)
                ax.annotate(util.round_special_no_zero_neg(corr[i],1), xy=(corr_x, 0.5), xycoords='axes fraction', fontsize=corr_fontsize,
                            horizontalalignment='left', verticalalignment='center',weight='bold',bbox=corr_bbox_props)
            if groups:
                if i==0:
                    ax.annotate(groups[0], xy=(groups_x, 0.05), xycoords='axes fraction', fontsize=group_fontsize,
                        horizontalalignment='left', verticalalignment='center',rotation = groups_rot)
                if i==2:
                    ax.annotate(groups[1], xy=(groups_x, 0), xycoords='axes fraction', fontsize=group_fontsize,
                        horizontalalignment='left', verticalalignment='center',rotation = groups_rot)

            i += 1


    axs[-1].set_xlabel('PMF',fontsize = fontsize)

    axs[-1].set_xticks([.5+(.1*i) for i in range(6)])
    axs[-1].set_xticklabels([str(round(.5+(.1*i),1)) for i in range(6)])
    axs[-1].tick_params(axis='both', labelsize=ticklabel_fontsize)
    # axs[-1].set_yticks(yticks[-1])
    axs[-1].set_xlim(min(regions[0]),max(regions[-1]))
    plt.subplots_adjust(hspace=0.0)
    if save:
        plt.savefig(save,dpi=300,bbox_inches = 'tight',pad_inches = 0.1)
    if show_fig:
        plt.show()
    else:
        plt.close()


def get_folder_data(folder,
                    container_folder = 'painting_preds',
                    notes = '',
                    img_path = 'extra_pollocks/Processed/Raw/',
                    filter_results= False,
                    save_folder = False,
                    r_name = 'r_data.csv',
                    results_name='results_data.csv',
                    load_results = False,
                    classification_thresh = 0.56,
                    one_vote = True,
                    update = False):
    if update:
        if os.path.exists(os.path.join(container_folder,folder,'results.csv')) and os.path.exists(os.path.join(container_folder,folder,'metrics.csv')) and os.path.exists(os.path.join(container_folder,folder,'full_paints_results.csv')):
            temp_results = pd.read_csv(os.path.join(container_folder,folder,'results.csv'))
            files = list(set(temp_results.painting))
            folders = util.os_listdir(os.path.join(container_folder,folder))
            if len(files) < len(folders) -3:
                results = utils_fastai.get_combined_dfs(folder,container_folder = container_folder,notes = notes,img_path = img_path)
            else:
                results = pd.read_csv(os.path.join(container_folder,folder,'results.csv'))        
        else:
            results = utils_fastai.get_combined_dfs(folder,container_folder = container_folder,notes = notes,img_path = img_path)
    else:
        if load_results:
            results = pd.read_csv(os.path.join(container_folder,folder,'results.csv'))
        else:
            results = utils_fastai.get_combined_dfs(folder,container_folder = container_folder,notes = notes,img_path = img_path)
    if filter_results: #filter results should be a list of painting names if not False
        results = results[results.painting_name.isin(filter_results)]
    binarize_prob = 0.5
    r = util.vote_system(results, one_vote=one_vote, binarize=False, binarize_prob=binarize_prob, decision_prob=classification_thresh)
    if save_folder:
        r.to_csv(os.path.join(save_folder,r_name))
        results.to_csv(os.path.join(save_folder,results_name))
    return r,results

def concat_images_vertically(images):
    # Get the size of the first image in the list
    width, height = images[0].size

    # Determine the width and height of the concatenated image
    max_width = width
    total_height = height * len(images)

    # Create a new image with the determined size
    concat_image = Image.new('RGB', (max_width, total_height))

    # Paste the images one below the other
    y_offset = 0
    for image in images:
        concat_image.paste(image, (0, y_offset))
        y_offset += height

    return concat_image

def plot_comparison_res(r_res,
                        threshold=0.56,
                        include_1 = False,
                        legend = ['P','J'],
                        selected_paths = False,
                        slice_sizes = False,
                        save = False,
                        show_fig = True,
                        factors = [1] + list(range(2,22,2))+[50,100]
                        ):
    
    r_res['factor']= [int(item.split('_')[-1]) for item in r_res.painting]
    r_res = r_res[r_res.factor.isin(factors)]
    r_P_sorted = r_res[r_res.painting.str.startswith('P')]
    r_J_sorted = r_res[r_res.painting.str.startswith('J')]
    r_P_means = r_P_sorted.groupby('factor').mean()
    r_J_means = r_J_sorted.groupby('factor').mean()
    P_factors = [str(item) for item in r_P_means.index]
    J_factors = [str(item) for item in r_J_means.index]
    P_means = r_P_means.pollock_prob.tolist()
    J_means = r_J_means.pollock_prob.tolist()

    if isinstance(include_1,pd.DataFrame):
        P_factors = ['1'] + P_factors
        J_factors = ['1'] + J_factors
        P_means = [include_1[include_1.painting.str.startswith('P')].pollock_prob.mean()] + P_means
        J_means = [include_1[include_1.painting.str.startswith('J')].pollock_prob.mean()] + J_means

    fig = plt.figure(figsize=(8, 10), facecolor='white')
    gs = fig.add_gridspec(3, 5, height_ratios=[1,0, 1], width_ratios=[.25,.25,.25,.25,.25])

    ax = fig.add_subplot(gs[2, :], facecolor='white')


    # plt.figure(facecolor='white')
    ax.plot(P_factors, P_means)
    ax.plot(J_factors, J_means)
    ax.axhline(y=threshold,linestyle = ':',color='black')

    # Thumbnail axis (ax_thumbs)
    locations = [1,3]
    rect_locations = [3.5,8.5]
    axes_locations = [0,10.2]
    if selected_paths:
        ax_thumbs = []
        # im_stack = []
        for i, paths in enumerate(selected_paths):
            images = [Image.open(path) for path in paths]
            im_stack = concat_images_vertically(images)
            ax_thumbs.append(fig.add_subplot(gs[0, locations[i]], facecolor='white'))
            ax_thumbs[i].axis('off')
            ax_thumbs[i].imshow(im_stack)

            # Add a box around the axes
            box = Rectangle((0, 0), 1, 1, transform=ax_thumbs[i].transAxes, linewidth=5, edgecolor='black', facecolor='none')
            ax_thumbs[i].add_patch(box)

            # Draw lines from the bottom of each rectangle to X='1' and X='20'
            x_values = [1, 20]
            y_bottom = 0  # The Y-coordinate of the bottom of the rectangle
            arrowprops = dict(arrowstyle='-', linewidth=1, color='black')
            for x in x_values:
                ax.annotate('', xy=(axes_locations[i], 1), xytext=(rect_locations[i], 1.12), xycoords=('data', 'axes fraction'), ha='center', va='center', fontsize=16,
                            arrowprops=arrowprops)

            # arrowprops = dict(arrowstyle='->',linewidth=2.5,color = 'k')
            # ax.annotate('', xy=(1, 2),xytext =(1,2), xycoords=('axes fraction', 'axes fraction'), ha='left', va='center', color='black', fontsize=16,
            #         arrowprops = arrowprops)
            # if slice_sizes:
            #     ax_thumbs[it].set_title(slice_sizes[i])
            

    if legend:
        ax.legend(legend)
    ax.set_yticks(np.arange(0.0,1.05,.2))
    ax.set_ylim(0,1)
    ax.set_xlim(0,12)
    ax.set_ylabel('PMF')
    ax.set_xlabel('Resolution Fraction')
    ax.set_xticks(np.arange(len(P_factors)))
    ax.set_xticklabels(['1/'+ i if int(i) >1 else '1' for i in P_factors ])
    
    plt.subplots_adjust(bottom=0.3, left=0.2, hspace=0, wspace=0) 
    if save:
        fig.savefig(save,dpi=300, bbox_inches="tight")
    if show_fig:
        fig.show()
    else:
        fig.close()
# def plot_comparison_res(r_res,threshold=0.56,include_1 = False,legend = ['P','J'],save = False,show_fig = True):
    
#     r_res['factor']= [int(item.split('_')[-1]) for item in r_res.painting]
#     r_P_sorted = r_res[r_res.painting.str.startswith('P')]
#     r_J_sorted = r_res[r_res.painting.str.startswith('J')]
#     r_P_means = r_P_sorted.groupby('factor').mean()
#     r_J_means = r_J_sorted.groupby('factor').mean()
#     P_factors = [str(item) for item in r_P_means.index]
#     J_factors = [str(item) for item in r_J_means.index]
#     P_means = r_P_means.pollock_prob.tolist()
#     J_means = r_J_means.pollock_prob.tolist()

#     if isinstance(include_1,pd.DataFrame):
#         P_factors = ['1'] + P_factors
#         J_factors = ['1'] + J_factors
#         P_means = [include_1[include_1.painting.str.startswith('P')].pollock_prob.mean()] + P_means
#         J_means = [include_1[include_1.painting.str.startswith('J')].pollock_prob.mean()] + J_means
#     plt.figure(facecolor='white')
#     plt.plot(P_factors, P_means)
#     plt.plot(J_factors, J_means)
#     plt.axhline(y=threshold,linestyle = ':',color='black')
#     if legend:
#         plt.legend(legend)
#     plt.yticks(np.arange(0.0,1.05,.2))
#     plt.ylim(0,1)
#     plt.xlim(0,12)
#     plt.ylabel('PMF')
#     plt.xlabel('Resolution Factor')
#     if save:
#         plt.savefig(save,dpi=300)
#     if show_fig:
#         plt.show()
#     else:
#         plt.close()
    

def plot_individual_res(r, column = 'painting',split_on = '_1',split_element = 0,figsize = (14,10),group = ('P','J'),sort = 'factor',include_1 = False):
    r_res = r[r.painting.str.startswith(group)].copy()
    paintings_res = util.get_base_file_name(r_res,column = column,split_on = split_on,split_element = split_element)
    r_res[sort]= [int(item.split('_')[-1]) for item in r_res.painting]
    plt.figure(figsize = figsize)
    for painting in paintings_res:
        one_painting = r_res[r_res.painting.str.startswith(painting + '_')].sort_values(sort)
        painting_factors = [str(item) for item in one_painting.factor]
        painting_values = one_painting.pollock_prob.to_list()
        if isinstance(include_1,pd.DataFrame):
            # print(painting_factors,painting_values)
            painting_factors = ['1'] + painting_factors
            painting_values = [include_1[include_1.painting == painting].pollock_prob.iloc[0]] + painting_values
            # print(painting_factors,painting_values)
        plt.plot(painting_factors,painting_values)
    # plt.grid(True)
    # plt.legend(paintings_res)
    plt.show()

def get_comparison_r(r,
                     master = False,
                     r2 = False,
                     left_suffix = 'Aug',
                     right_suffix = 'Orig',
                     sort = 'abs_diff',
                     learner_folder = 'gcp_classifier-3_10-Max_color_10-15-2022',
                     print_stats = True,
                     master_columns = ['file','catalog','set','artist','group','is_pollock']):
    # example: conditions = ['artist]
    if not isinstance(master,pd.DataFrame):
        master = pd.read_parquet('master.parquet')
    if not isinstance(r2,pd.DataFrame):
        results2 = utils_fastai.get_learner_results(learner_folder=learner_folder, sets = ('train','valid','hold_out')) 
        r2 = util.vote_system(results2)
    r = pd.merge(left = r[['painting','pollock_prob','failed']], right = r2[['painting','pollock_prob','failed']],how = 'left', on = 'painting',suffixes=('_'+left_suffix,'_'+right_suffix))
    r = pd.merge(left = r, right = master[master_columns],left_on = 'painting',right_on = 'file',how='left')
    r['diff'] = r['pollock_prob_' + left_suffix] - r['pollock_prob_' + right_suffix]
    r['abs_diff'] = r['diff'].abs()
    if sort:
        r.sort_values(sort,ascending=False,inplace = True)
    r.reset_index(drop = 'index',inplace = True)
    if print_stats:
        print_comparison_r_stats(r,left_suffix = left_suffix,right_suffix = right_suffix)
    return r

def print_comparison_r_stats(r,left_suffix = 'Aug',right_suffix = 'Orig'):
    print('number of files (Pollock,non-pollock)','(' + str(len(r[r.artist == 'Pollock']))+','+str(len(r[r.artist != 'Pollock']))+')')
    print('diff',r['diff'].mean())
    print('abs_diff',r['abs_diff'].mean())
    print(left_suffix + '_failed', len(r[r['failed_' + left_suffix]]))
    print(right_suffix + '_failed',len(r[r['failed_' + right_suffix]]))

# def get_thesh_vs_MA(results,master,
#                     round_to = 2,
#                     remove_special = ['P69(V)','P43(W)','JPCR_01031','A9(right)','JPCR_01088','P13(L)','P19(L)','P25(L)','P32(W)','P33(W Lowerqual)','P47(F Lowres)','P65(L)','P75(V)','P77(J)','P80(J)','P86(S)','P105(J)','P106(V)','P115(F Lowres)','A14(Left)'],
#                     threshs = np.arange(0,1.01,0.01),
#                     save = False,
#                     ):
#     MA = []
#     for thresh in threshs:
#         machine_accuracy_images = rubric.get_machine_accuracy(results,master,
#                                                             remove_special = remove_special,
#                                                             catagory = ('J','P'),
#                                                             one_vote = True,
#                                                             use_images = True,
#                                                             percent = round_to,
#                                                             binarize = False,
#                                                             binarize_prob=0.5,
#                                                             decision_prob=thresh)
#         # machine_accuracy_images = str_round(machine_accuracy_images,round_to)
#         MA.append(machine_accuracy_images)
#     master = master[(~master.remove_special) & (master.artist != 'unknown') & (master.set != 'train')]
#     r = util.vote_system(results)
#     R = pd.merge(left = r, right = master[['file','set','is_pollock']], left_on ='painting',right_on = 'file')
#     df = R.copy()
#     bin_width = 0.1
#     bins = [x for x in np.arange(0, 1.1, bin_width)]

#     # Separate data for 'True' and 'False' is_pollock values
#     is_pollock_true = df[df['is_pollock'] == 'True']['pollock_prob']
#     is_pollock_false = df[df['is_pollock'] == 'False']['pollock_prob']

#     # Set custom colors for each condition
#     color_true = [0,0.5,0]
#     color_false = [1,0,0]



#     fig = plt.figure(figsize=(10,6),facecolor = 'white')
#     plt.plot(threshs, MA,color = 'black')
#     # Create the histograms for both conditions
#     plt.hist([is_pollock_true, is_pollock_false], bins=bins, edgecolor='black', alpha=1, color=[color_true, color_false], stacked=True, label=['Pollock', 'non-Pollock'])
#     max_index = np.argmax(MA)
#     # Add a vertical line at the position of the maximum value
#     plt.axvline(x=threshs[max_index+1], color='r', linestyle='--')
#     plt.xticks(np.append(np.arange(0,1.1,0.1),threshs[max_index+1]),rotation = 90)
#     plt.xlabel('Clasification Threshold')
#     plt.ylabel('Machine Accuracy (%)')
#     if save:
#         plt.savefig(save, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()
#     return np.max(MA)
def get_thesh_vs_MA(results, master, round_to=2, catagory=('J', 'P'),remove_special=['P69(V)', 'P43(W)', 'JPCR_01031', 'A9(right)', 'JPCR_01088', 'P13(L)', 'P19(L)', 'P25(L)', 'P32(W)', 'P33(W Lowerqual)', 'P47(F Lowres)', 'P65(L)', 'P75(V)', 'P77(J)', 'P80(J)', 'P86(S)', 'P105(J)', 'P106(V)', 'P115(F Lowres)', 'A14(Left)'], threshs=np.arange(0, 1.01, 0.01), save=False):
    MA = []
    for thresh in threshs:
        machine_accuracy_images = rubric.get_machine_accuracy(results, master, remove_special=remove_special, catagory=catagory, one_vote=True, use_images=True, percent=round_to, binarize=False, binarize_prob=0.5, decision_prob=thresh)
        MA.append(machine_accuracy_images)
    
    master = master[(~master.remove_special) & (master.artist != 'unknown') & (master.set != 'train')]
    r = util.vote_system(results)
    R = pd.merge(left=r, right=master[['file', 'set', 'is_pollock']], left_on='painting', right_on='file')
    df = R.copy()
    bin_width = 0.1
    bins = [x for x in np.arange(0, 1.1, bin_width)]
    
    is_pollock_true = df[df['is_pollock'] == 'True']['pollock_prob']
    is_pollock_false = df[df['is_pollock'] == 'False']['pollock_prob']
    
    color_true = [0, 0.5, 0]
    color_false = [1, 0, 0]
    
    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor='white')
    
    
    max_index = np.argmax(MA)
    
    all_threshs = np.append(np.arange(0, 1.1, 0.1), threshs[max_index + 1])
    ls = 12
    fs = 20
    plt.hist([is_pollock_true, is_pollock_false], bins=bins, edgecolor='black', alpha=1, color=[color_true, color_false], stacked=True, label=['Pollock', 'non-Pollock'])
    ax1.set_xticks(all_threshs)
    ax1.set_xticklabels([round(T,2) for T in all_threshs], rotation=90)
    ax1.xaxis.set_tick_params(labelsize=ls) 
    ax1.set_xlabel('PMF', fontsize=fs)
    ax1.set_ylabel('n', fontsize=fs)
    ax1.yaxis.set_tick_params(labelsize=ls) 
    # ax1.legend(loc='upper left')
    
    # Create a second y-axis
    ax2 = ax1.twinx()
    ax2.plot(threshs, MA, color='black', label='Machine Accuracy',linewidth = 3)
    ax2.axvline(x=threshs[max_index + 1], color='k', linestyle='--', label='Max Accuracy Threshold')
    ax2.set_ylabel('MA (%)', fontsize=fs)  # Label for the second y-axis
    ax2.spines['left'].set_linewidth(2)  # Increase linewidth of the left y-axis border
    ax2.spines['right'].set_linewidth(2)  # Increase linewidth of the right y-axis border
    ax2.spines['top'].set_linewidth(2)  # Increase linewidth of the left y-axis border
    ax2.spines['bottom'].set_linewidth(2)  # Increase linewidth of the right y-axis border
    ax2.set_ylim([0,100])
    ax2.yaxis.set_tick_params(labelsize=ls) 
    
    
    
    if save:
        plt.savefig(save, bbox_inches='tight', dpi = 300)
        plt.close()
    else:
        plt.show()
    
    return np.max(MA),threshs[np.argmax(MA)]

def plot_BrightnessContrast(DF,
                            fixed_value = 0,
                            fixed_type = 'Contrast',
                            set_xlim = False,
                            select_files = False, 
                            title = False,
                            save = False,
                            show_plot = True,
                            thresh = 0.56,
                            cat_dict = {'P':'P','J':'J','A':'A','C':'C','D':'D','E':'E', 'G':'G','F':'F'}
                            ):
    df = DF.copy()
    
    df['group'] = df['group'].map(cat_dict)

    if fixed_type == 'Contrast':
        variable_type = 'Brightness'
    elif fixed_type == 'Brightness':
        variable_type = 'Contrast'
    else:
        print('aint no type for that, check your fixed type')
    if select_files:
        assert isinstance(select_files,list), 'select files must be list'
        df = df[df['painting'].isin(select_files)]
    filtered_df = df[df[fixed_type] == fixed_value]

    # Define custom colors for each group
    custom_palette = {'A': 'red', 'C': 'blue', 'D': 'green', 'E': 'purple',
                    'F': 'orange', 'G': 'pink', 'J': 'brown', 'P': 'cyan','H': 'black','Pollock':'green','not-Pollock':'red'}

    # Plot using seaborn with custom palette
    # sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6),facecolor='white')
    ax = sns.lineplot(x=variable_type, y='pollock_prob_Aug', hue='group', ci='sd', data=filtered_df, palette=custom_palette)
    plt.axhline(y=thresh, color='black', linestyle='--')

    # Get handles and labels for the legend
    handles, labels = ax.get_legend_handles_labels()
    # handles_with_sd = []

    # for i in range(len(handles)):
    #     handle_mean = plt.Line2D([0], [0], color='w', marker='o', markersize=10, label=labels[i])
    #     handle_sd = plt.Line2D([0], [0], color='w', marker='o', markersize=10, markerfacecolor='black', label=f'SD={filtered_df[filtered_df["group"] == labels[i]]["pollock_prob_Aug"].std():.2f}')
    #     handles_with_sd.extend([handle_mean, handle_sd])


    # plt.legend(handles=handles_with_sd)

    # Sort the legend labels alphabetically
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))

    # Plot the legend with alphabetical order
    plt.legend(handles, labels)

    if title:
        plt.title(title)
    else:
        plt.title('pollock_prob_Aug for each group at ' + fixed_type + f' = {fixed_value}')
    plt.xlabel(variable_type, fontsize=16)
    plt.ylabel('PMF', fontsize=16)
    plt.ylim([-0.05,1.05])
    
    # Access the current Axes object
    ax = plt.gca()
    # Set the linewidth of the spines (outer box lines)
    ax.spines['top'].set_linewidth(2)    # Top border
    ax.spines['bottom'].set_linewidth(2) # Bottom border
    ax.spines['left'].set_linewidth(2)   # Left border
    ax.spines['right'].set_linewidth(2)  # Right border

    if isinstance(set_xlim,list):
        plt.xlim(set_xlim)
    if save:
        if not os.path.exists(Path(save).parent):
            os.makedirs(Path(save).parent)
        plt.savefig(save,dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()
import rubric

def get_3_panel_confidence(results, master, round_to=2, remove_special=['P69(V)', 'P43(W)', 'JPCR_01031', 'A9(right)', 'JPCR_01088', 'P13(L)', 'P19(L)', 'P25(L)', 'P32(W)', 'P33(W Lowerqual)', 'P47(F Lowres)', 'P65(L)', 'P75(V)', 'P77(J)', 'P80(J)', 'P86(S)', 'P105(J)', 'P106(V)', 'P115(F Lowres)', 'A14(Left)'], threshs=np.arange(0, 1.01, 0.01), save=False):
    MA = []
    for thresh in threshs:
        machine_accuracy_images = rubric.get_machine_accuracy(results, master, remove_special=remove_special, catagory=('J', 'P'), one_vote=True, use_images=True, percent=round_to, binarize=False, binarize_prob=0.5, decision_prob=thresh)
        MA.append(machine_accuracy_images)
    
    master = master[(~master.remove_special) & (master.artist != 'unknown') & (master.set != 'train')]
    r = util.vote_system(results)
    R = pd.merge(left=r, right=master[['file', 'set', 'is_pollock']], left_on='painting', right_on='file')
    df = R.copy()
    bin_width = 0.1
    bins = [x for x in np.arange(0, 1.1, bin_width)]
    
    is_pollock_true = df[df['is_pollock'] == 'True']['pollock_prob']
    is_pollock_false = df[df['is_pollock'] == 'False']['pollock_prob']
    
    color_true = [0, 0.5, 0]
    color_false = [1, 0, 0]
    
    # fig, ax1 = plt.subplots(figsize=(10, 6), facecolor='white')
    
    
    max_index = np.argmax(MA)
    
    all_threshs = np.append(np.arange(0, 1.1, 0.1), threshs[max_index + 1])
    ls = 12
    fs = 20

    # Compute the histogram for 'Pollock' class
    hist_pollock, _ = np.histogram(is_pollock_true, bins=bins)
    PP = hist_pollock/sum(hist_pollock)
    
    # Compute the histogram for 'non-Pollock' class
    hist_non_pollock, _ = np.histogram(is_pollock_false, bins=bins)
    PN = hist_non_pollock/sum(hist_non_pollock)
    CP = PP/(PP+PN)
    # CP = np.nan_to_num(CP, nan=0)
    CNP = PN/(PP+PN)
    # CNP = np.nan_to_num(CNP, nan=0)

    # Create a figure and a 3x1 grid of subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), facecolor='white', sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [1, 1, 1]})



    # Plot the histograms in the bottom subplot
    axs[2].hist([is_pollock_true, is_pollock_false], bins=bins, edgecolor='black', alpha=1, color=[color_true, color_false], stacked=True, label=['Pollock', 'non-Pollock'])
    axs[2].set_xticks(all_threshs)
    axs[2].set_xticklabels([round(T,2) for T in all_threshs], rotation=90)
    axs[2].xaxis.set_tick_params(labelsize=ls) 
    axs[2].axvline(x=threshs[max_index + 1], color='r', linestyle='--', label='Max Accuracy Threshold')
    axs[2].set_xlabel('PMF', fontsize=fs)
    axs[2].set_ylabel('n', fontsize=fs)
    axs[2].yaxis.set_tick_params(labelsize=ls) 

    # Plot the line plot in the middle subplot
    axs[0].plot(threshs, MA, color='black', label='Machine Accuracy', linewidth=3)
    axs[0].axvline(x=threshs[max_index + 1], color='r', linestyle='--', label='Max Accuracy Threshold')
    axs[0].set_ylabel('Machine Accuracy (%)', fontsize=fs)
    axs[0].set_ylim([0, 100])
    axs[0].yaxis.set_tick_params(labelsize=ls) 

    # Plot the line plot of CP and CNP in the top subplot
    # Replace 'cp' and 'cnp' with your actual data
    # Calculate the midpoints of the bins
    thresh_midpoints = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
    axs[1].plot(thresh_midpoints, CP, color='green', label='CP', linewidth=3)
    axs[1].plot(thresh_midpoints, CNP, color='red', label='CNP', linewidth=3)
    axs[1].axvline(x=threshs[max_index + 1], color='r', linestyle='--', label='Max Accuracy Threshold')
    axs[1].set_ylabel('Confidence', fontsize=fs)  # Label for the top subplot
    axs[1].yaxis.set_tick_params(labelsize=ls) 
    # axs[1].legend()

    # Additional customizations as needed

    # Show or save the figure
    if save:
        plt.savefig(save, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_BCHS(df,variable_type = 'Brightness',default_dict = {'Brightness':1,'Contrast':1,'Hue':0,'Saturation':1},set_xlim = False,select_files = False, title = False,save = False,show_plot = True):
    #
 
    if select_files:
        assert isinstance(select_files,list), 'select files must be list'
        df = df[df['painting'].isin(select_files)]
    for key in default_dict:
        if key != variable_type:
            df = df[df[key] == default_dict[key]]

    # Define custom colors for each group
    custom_palette = {'A': 'red', 'C': 'blue', 'D': 'green', 'E': 'purple',
                    'F': 'orange', 'G': 'pink', 'J': 'brown', 'P': 'cyan','H': 'black'}

    # Plot using seaborn with custom palette
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6),facecolor='white')
    ax = sns.lineplot(x=variable_type, y='pollock_prob_Aug', hue='group', ci=None, data=df, palette=custom_palette)

    # Get handles and labels for the legend
    handles, labels = ax.get_legend_handles_labels()

    # Sort the legend labels alphabetically
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))

    # Plot the legend with alphabetical order
    plt.legend(handles, labels, title='Group')

    if title:
        plt.title(title)
    else:
        plt.title('pollock_prob_Aug for each group')
    plt.xlabel(variable_type)
    plt.ylabel('Mean pollock_prob_Aug')
    plt.ylim([-0.05,1.05])
    if isinstance(set_xlim,list):
        plt.xlim(set_xlim)
    if save:
        plt.savefig(save)
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_individual_contrast(df,file,fixed_value = 0,fixed_type = 'Contrast',save = False):
    if fixed_type == 'Contrast':
        variable_type = 'Brightness'
    elif fixed_type == 'Brightness':
        variable_type = 'Contrast'
    else:
        print('aint no type for that, check your fixed type')
    df = df[df[fixed_type] == fixed_value]
    target_df = df[df['painting']==file]

    # Sort the DataFrame by Contrast in ascending order
    sorted_df = target_df.sort_values(by=variable_type)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_df[variable_type], sorted_df['pollock_prob_Aug'], marker='o')

    # Add labels and title
    plt.xlabel(variable_type)
    plt.ylabel('pollock_prob_Aug')
    plt.title(f'Brightness/Contrast Plot for {file}')

    # Show the plot
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def get_BrightnessContrast_df(full_r,learner_master,
                              file_start = 'RBC_B',
                              path = 'painting_preds',
                              standard_filter = True,
                              update = True,
                              master_columns = ['file','catalog','set','artist','group','is_pollock']): 
    # file_start = 'RBC_B'
    # file_start = 'C14_Bpt'
    files = [file for file in util.os_listdir(path) if file.startswith(file_start)]
    if standard_filter:
        learner_master = learner_master[~learner_master.group.isin(['B']) & learner_master.set.isin(('valid','hold_out')) & ~learner_master.remove_special]   
    r = {}
    R = {}
    for file in files:
        if util.check_file_in_subfolders(os.path.join(path,file), 'results.csv'):
            # print(file)
            r[file] , _ = get_folder_data(file,container_folder = path,notes = '',img_path = 'Paintings/Processed/Raw/',update =  update,load_results=False)
            R[file] = get_comparison_r(r[file],r2 = full_r,left_suffix = 'Aug',print_stats=False,master = learner_master,master_columns = master_columns)
            # if (len(R[file][~R[file].painting.str.startswith(('P','J'))]) > 0)  & (len(R[file][R[file].painting.str.startswith(('P','J'))]) > 0):
            R[file]['Brightness'] = float(file.split('_Bpt')[1].split('_')[0])/100
            R[file]['Contrast'] = float(file.split('_Cpt')[1].split('_')[0])/100
    df = pd.concat(R.values(), ignore_index=True)
    return df

def get_BCHS_df(full_r,learner_master,file_start = 'RBC_B',path = 'painting_preds',standard_filter = True,master_columns = ['file','catalog','set','artist','group','is_pollock']): 
    # file_start = 'RBC_B'
    # file_start = 'C14_Bpt'
    files = [file for file in util.os_listdir(path) if file.startswith(file_start)]
    if standard_filter:
        learner_master = learner_master[~learner_master.group.isin(['B']) & learner_master.set.isin(('valid','hold_out')) & ~learner_master.remove_special]   
    r = {}
    R = {}
    for file in files:
        if util.check_file_in_subfolders(os.path.join(path,file), 'results.csv'):
            r[file] , _ = get_folder_data(file,container_folder = path,notes = '',img_path = 'Paintings/Processed/Raw/',update =  True,load_results=False)
            R[file] = get_comparison_r(r[file],r2 = full_r,left_suffix = 'Aug',print_stats=False,master = learner_master,master_columns = master_columns)
            # if (len(R[file][~R[file].painting.str.startswith(('P','J'))]) > 0)  & (len(R[file][R[file].painting.str.startswith(('P','J'))]) > 0):
            R[file]['Brightness'] = float(file.split('_')[1].split('pt')[1])/100
            R[file]['Contrast'] = float(file.split('_')[2].split('pt')[1])/100
            R[file]['Hue'] = float(file.split('_')[3].split('pt')[1])/100
            R[file]['Saturation'] = float(file.split('_')[4].split('pt')[1])/100
    df = pd.concat(R.values(), ignore_index=True)
    return df

def get_painting_preds_df(master,
                            img_path = 'Paintings/Processed/Raw/',
                            file_start = 'RBC_B',
                            path = 'painting_preds',
                            update = True,
                            sort = 'pollock_prob',
                            master_columns = ['file','catalog','set','artist','group','is_pollock'],
                            value_dict = {'Brightness':'default','Contrast':'default'}): 
    # file_start = 'RBC_B'
    # file_start = 'C14_Bpt'
    files = [file for file in util.os_listdir(path) if file.startswith(file_start)]
    r = {}
    R = {}
    for file in files:
        if util.check_file_in_subfolders(os.path.join(path,file), 'results.csv'):
            # print(file)
            r[file] , _ = get_folder_data(file,container_folder = path,notes = '',img_path = img_path,update =  update,load_results=False)
            R[file] = pd.merge(left = r[file][['painting','pollock_prob','failed']], right = master[master_columns],left_on = 'painting',right_on = 'file',how='left')
            for item in value_dict:
                R[file][item] = make_column_value_from_file_name(file,key = item, value= value_dict[item])
    df = pd.concat(R.values(), ignore_index=True)
    if sort:
        df.sort_values(sort,ascending=False,inplace = True)
    df.reset_index(drop = 'index',inplace = True)
    return df

def svelt_r(r,master,
            r_columns = ['painting','pollock_prob','failed'],
            master_columns = ['file','catalog','set','title','artist','group','is_pollock','remove_special']):
    return pd.merge(left = r[r_columns], right = master[master_columns],left_on = 'painting',right_on = 'file',how='left')



def make_column_value_from_file_name(file,key = 'Brightness', value= 'default'):
    default_dict = {'Brightness':"float(file.split('_Bpt')[1].split('_')[0])/100",
                    'Contrast':"float(file.split('_Cpt')[1].split('_')[0])/100",
                    'res':"float(file.split('DS_')[1].split('_')[0])"
                    }
    if value == 'default':
        return eval(default_dict[key])
    else:
        return eval(value)


def get_r_master(model_folder = 'gcp_classifier-3_10-Max_color_10-15-2022',master_file = 'master.parquet',sets = ('train','valid','hold_out')):
    master = pd.read_parquet(master_file)
    results = utils_fastai.get_learner_results(learner_folder=model_folder,sets = sets)
    r = util.vote_system(results)
    return r,master