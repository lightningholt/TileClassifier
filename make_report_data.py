#make_report_data.py

import pandas as pd
import util
import utils_fastai
import plotting
from pathlib import Path
import os
from PIL import Image,ImageOps
import numpy as np
import rubric
import math
import stats
import visualizer as vz
import cv2
from util import str_round
import albumentations as A
import csv

def make_report_data(master,learner_master, 
                     container = 'runs',
                     learner_folder = 'gcp_classifier-3_10-Max_color_10-15-2022',
                     img_path_testing =  'Bristol/Processed/Raw',
                     img_path =  'Paintings/Processed/Raw',
                     save_pre_path = 'painting_preds',
                     save_folder = 'new_run',
                     do_preds = False, #test painting dependent (i dont think)
                     do_stats = False, #NOT test painting dependent (i dont think)
                     do_viz = False, #test painting dependent (i dont think)
                     do_comparison_viz = False, #NOT test painting dependent (i dont think)
                     skip_viz_data = False, #Applies to both test and not test
                     do_viz_tiling = True,
                     write_individual_layers = False,
                     do_counts = False, #NOT test painting dependent (i dont think)
                     do_orientation_confirmation = False, #test painting dependent
                     round_to = 2,
                     round_to_acc = 2,
                     one_vote=True,
                     remove_special = ['P69(V)','P43(W)','JPCR_01031','A9(right)','JPCR_01088','P13(L)','P19(L)','P25(L)','P32(W)','P33(W Lowerqual)','P47(F Lowres)','P65(L)','P75(V)','P77(J)','P80(J)','P86(S)','P105(J)','P106(V)','P115(F Lowres)','A14(Left)'], #'JPCR_01030','P42(W)',
                     classification_thresh = 0.56,
                     select_painting = False,
                     select_painting_map = False,
                     select_painting_2 = False,
                     additional_dial_paintings = ['D15','C64','P4(F)','P86(S)','P123(W)'],
                     make_report=True,
                     verbose = True,
                     add = 5,
                     dummy = False,
                     item_tfms = False,
                     do_res_tile = False,
                     res_tile_cm = 20,
                     res_factor = 30,
                     res_folder = 'res_standard',
                     resize = (500,500),
                     pollock_categories = ('P','J'),
                     np_categories = ('A','C','D','E','F','G'),
                     pollock_viz_path_start = 'viz/PJ',
                     np_viz_path_start = 'viz/NP_overlap',
                     dial_paintings = ['F22','D15', 'C2', 'F89', 'C45', 'P2(V)', 'P4(F)', 'C63', 'F65', 'JPCR_00173','P68(V)'] #'C2','A66','C53','A12',
                     
                    ):
    sub_dir= 'figs'
    stats_sub_dir = 'stats'    
    
    if not os.path.exists(os.path.join(save_pre_path,save_folder,sub_dir)):
        os.makedirs(os.path.join(save_pre_path,save_folder,sub_dir))
    ######
    #load in main master
    # M = learner_master

    learner_metrics = pd.read_csv(Path(container,learner_folder,'metrics.csv'))
    # learner_results = pd.read_csv(Path(container,learner_folder,'results.csv'))
    # learner_results_ho = pd.read_csv(Path(container,learner_folder,'preds','HO_raw','results.csv'))
    # # print('learner_results',len(learner_results))
    # # print('learner_results_ho',len(learner_results_ho))
    learner_results = utils_fastai.get_learner_results(learner_folder=learner_folder,sets = ('valid','hold_out')) 

    # learner_results_special_removed  = learner_results[~learner_results.file.str.startswith(tuple(remove_special))].reset_index() #removes hold out special
    
    #get rid of remove_special (hold_out_duplicates and those few files accidentally included)
    # filter_away_items = [item + '_c' for item in remove_special]
    # learner_results=learner_results[~learner_results.file.str.startswith(tuple(filter_away_items))]    
    # print('learner_results',len(learner_results))
    folders = learner_metrics.folders.apply(eval).iloc[0]
    full_results = utils_fastai.get_learner_results(learner_folder=learner_folder,sets = ('train','valid','hold_out')) 

    # full_results_special_removed  = full_results[~full_results.file.str.startswith(tuple(remove_special))].reset_index() #removes hold out special

    full_r = util.vote_system(full_results)
    # full_r_special_removed = util.vote_system(full_results)

    df = master.copy()
    title = df.file.iloc[0]   
    folder_str = title + '_' + 'overlap' + '_' + str(add)  
    if do_preds:
        utils_fastai.get_learn_preds(master,learner_folder,
                                     img_path = img_path_testing,
                                     folders = folders,
                                     learn_pre_path =container,
                                     save_pre_path = save_pre_path,
                                     write_name = save_folder,
                                     notes='',
                                     overwrite=True,
                                     multipath = False,
                                     verbose = verbose,
                                     item_tfms = item_tfms)
    sub_folder = 'individual_' + learner_folder
    # sub_folder = 'individual'
    if do_viz:
        group = title
        if skip_viz_data:
            fs = folder_str
            combined_result = np.load(os.path.join('viz',sub_folder,fs,fs,'combined.npy'))
            vz.pollock_map(combined_result,
                           hist_width = 0.4,
                           custom_cmap=True,
                           save_path=os.path.join('viz',sub_folder,fs,fs,'pollock_map.png'),
                           gradient = False,
                           round_to = round_to_acc)
        else:
            vz.process_image_wrapper(read_dir_full_cropped = os.path.join(img_path_testing,'Full'),
                                    testing_master = master,
                                    learner_folder=learner_folder,
                                    group = group,
                                    center_tile = False,
                                    specific_files_in_master = False, #should be list of files in testing_master (not the base file name) if not False
                                    verbose = False,
                                    do_tiling = do_viz_tiling,
                                    add = add,
                                    viz_dir = os.path.join('viz',sub_folder), #os.path.join(save_pre_path,save_folder,'viz'),
                                    painting_preds_dir = os.path.join('painting_preds',sub_folder), #os.path.join(save_pre_path,save_folder,'viz_preds'),
                                    specify_viz_save=False, #os.path.join(save_pre_path,save_folder,sub_dir,'select')
                                    resize = resize,
                                    item_tfms = item_tfms,
                                    write_individual_layers=write_individual_layers
                                    )
    if do_comparison_viz:
        group = select_painting_map
        if skip_viz_data:
            fs = group + '_' + 'overlap' + '_' + str(add)  
            combined_result = np.load(os.path.join('viz',sub_folder,fs,fs,'combined.npy'))
            vz.pollock_map(combined_result,
                           hist_width = 0.4,
                           custom_cmap=True,
                           save_path=os.path.join('viz',sub_folder,fs,fs,'pollock_map.png'),
                           gradient = False,
                           round_to = round_to_acc)
        else:
            vz.process_image_wrapper(read_dir_full_cropped = os.path.join(img_path,'Full'),
                                testing_master = learner_master,
                                learner_folder = learner_folder,
                                group = group,
                                center_tile = False,
                                specific_files_in_master = [group], #should be list of files in testing_master (not the base file name) if not False
                                verbose = False,
                                add = add,
                                do_tiling = do_viz_tiling,
                                viz_dir = os.path.join('viz',sub_folder), #os.path.join(save_pre_path,save_folder,'viz'),
                                painting_preds_dir = os.path.join('painting_preds',sub_folder), #os.path.join(save_pre_path,save_folder,'viz_preds'),
                                specify_viz_save=False #os.path.join(save_pre_path,save_folder,sub_dir,'select')
                                )
        group = select_painting_2
        if skip_viz_data:
            fs = group + '_' + 'overlap' + '_' + str(add)  
            combined_result = np.load(os.path.join('viz',sub_folder,fs,fs,'combined.npy'))
            vz.pollock_map(combined_result,
                           hist_width = 0.4,
                           custom_cmap=True,
                           save_path=os.path.join('viz',sub_folder,fs,fs,'pollock_map.png'),
                           gradient = False,
                           round_to = round_to_acc)
        else:
            vz.process_image_wrapper(read_dir_full_cropped = os.path.join(img_path,'Full'),
                                testing_master = learner_master,
                                learner_folder = learner_folder,
                                group = group,
                                center_tile = False,
                                specific_files_in_master = [group], #should be list of files in testing_master (not the base file name) if not False
                                verbose = False,
                                add = add,
                                do_tiling = do_viz_tiling,
                                viz_dir = os.path.join('viz',sub_folder),
                                painting_preds_dir = os.path.join('painting_preds',sub_folder),
                                specify_viz_save=False #os.path.join(save_pre_path,save_folder,sub_dir,'select_2')
                                )
    if do_stats:
        if not os.path.exists(os.path.join(container,learner_folder,stats_sub_dir)):
            os.makedirs(os.path.join(container,learner_folder,stats_sub_dir))
        scales = stats.get_scales(full_results,categories=False,add_max = True,verbose = False,save = os.path.join(container,learner_folder,stats_sub_dir,'scales.csv'))
        stats.get_folder_data(res_folder,
                              container_folder = 'painting_preds',
                              notes = '',
                              img_path = 'resolution/Processed/Raw/',
                              save_folder = os.path.join(container,learner_folder,stats_sub_dir),
                              r_name= 'r_res.csv',
                              results_name='results_res.csv'
                              )
        PJ = stats.get_stats(full_results,scales,learner_master,categories = pollock_categories,viz_path_start = pollock_viz_path_start,skip_viz=False,verbose =True,save = os.path.join(container,learner_folder,stats_sub_dir,'stats.csv'),do_year = True)
        NP = stats.get_stats(full_results,scales,learner_master,
                             categories = np_categories,
                             verbose =True,
                             save = os.path.join(container,learner_folder,stats_sub_dir,'stats_NP.csv'),
                             viz_path_start = np_viz_path_start,
                             viz_path_end  = 'overlap',
                             skip_viz=False,
                             do_year = True)
        stats.get_folder_data('AI_standard',container_folder = 'painting_preds',notes = '',img_path = 'Imitations_Summer_2023/Processed/Raw/')
        stats.get_folder_data('T2_standard',container_folder = 'painting_preds',notes = '',img_path = 'Test_Imitations-2/Processed/Raw')
        stats.get_folder_data('bw_standard',container_folder = 'painting_preds',notes = '',img_path = 'BW_converted/Processed/Raw/')
        stats.get_folder_data('henri_standard',container_folder = 'painting_preds',notes = '',img_path = 'Henri/Processed/Raw')
    if do_counts:
        print('getting counts')
        util.get_tile_counts(learner_master,
                  base_folder = img_path,
                  full_paint_folder = 'Full',
                  output_sizes_cm=range(10,365,5),
                  from_master = True,
                  verbose = False,
                  files_from_dir = False,
                  save = str(Path(Path(img_path).parts[0],'counts.csv'))
        )
    if do_res_tile:
        util.tile_res_specific(title,master,
                    output_size_cm = res_tile_cm,
                    res_factor = res_factor,
                    img_path_testing = img_path_testing,
                    verbose = False,
                    center_tile = True,
                    add = 0,
                    update = False
                    )
    tfms = ['Rotate:limit=(90,90)','Rotate:limit=(180,180)','Rotate:limit=(270,270)','HorizontalFlip','VerticalFlip']
    ttype = [ii.split(':')[0] for ii in tfms]
    args = [ii.split(')')[0].split(',')[-1] for ii in tfms]
    groups = [g + '_' + a if g != a else g for g, a in zip(ttype, args)]    
    orientation_path = os.path.join(save_pre_path,save_folder,'orientation')   
    if do_orientation_confirmation:
        util.print_if(verbose,'doing orientation confirmation')
        if not os.path.exists(orientation_path):
            os.makedirs(orientation_path)
        for group,tfm in zip(groups,tfms):
            item_tfms = utils_fastai.get_albumentations_tfms(tfm)
            vz.process_image_wrapper(read_dir_full_cropped = os.path.join(img_path_testing,'Full'),
                            testing_master = master,
                            learner_folder = learner_folder,
                            group = group,
                            center_tile = True,
                            specific_files_in_master = False, #should be list of files in testing_master (not the base file name) if not False
                            verbose = verbose,
                            add = 0,
                            painting_preds_dir = orientation_path,
                            item_tfms = item_tfms,
                            do_tiling = False
                            )
            stats.get_folder_data(group + '_standard',container_folder = orientation_path,notes = '',img_path = img_path_testing,load_results = False)
            stats.get_folder_data(group + '_standard',container_folder = 'painting_preds',notes = '',img_path = img_path,load_results = False)


    if make_report:
        if not select_painting_map:
            select_painting_map = select_painting
        util.copy_pollock_maps(source_dir=os.path.join('viz',sub_folder),
                               save_dir = os.path.join(save_pre_path,save_folder,sub_dir),
                               add = add,
                               paintings = [title,select_painting_map, select_painting_2],
                               painting_names = ['gray.png','overlay.png','pollock_map.png'])
        #For pollock dial
        
        test_ext = '_cropped_C0_R0.tif'
        ####

        #get orientation data
        rr = {}
        V_HO_poured_master = learner_master[~learner_master.group.isin(['B','F']) & learner_master.set.isin(('valid','hold_out')) & ~learner_master.remove_special]        
        r = {}
        for group in groups:
            r[group], _ = stats.get_folder_data(group + '_standard',container_folder = orientation_path,notes = '',img_path = img_path_testing,load_results = True)
            r_temp, _ = stats.get_folder_data(group + '_standard',container_folder = 'painting_preds',notes = '',img_path = img_path,load_results = True)
            R= stats.get_comparison_r(r_temp,r2 = full_r,left_suffix = 'Aug',print_stats=False,master = V_HO_poured_master)
            # stats.print_comparison_r_stats(R[(R.set != 'train')],left_suffix='180_rotation',right_suffix='Orig')
            rr[group] = R[(R.set != 'train')].groupby('is_pollock').mean().reset_index()[['is_pollock','pollock_prob_Aug','pollock_prob_Orig','diff']]

        if os.path.exists(Path(container,learner_folder,'summary.csv')):
            learner_summary = pd.read_csv(Path(container,learner_folder,'summary.csv'))
        else:
            learner_summary = rubric.get_summary_df(learner_folder,container=container,save = 'summary.csv')
        
        metrics = pd.read_csv(Path(save_pre_path,save_folder,'metrics.csv'))
        results = pd.read_csv(Path(save_pre_path,save_folder,'results.csv'))

        df = master.copy()
        title = df.file.iloc[0]
        title_testing = df.title.iloc[0]
        # print(title)
        submitted_by = df.source.iloc[0]
        canvas_size = str(df.height_cm.iloc[0]) +' x '+str(df.width_cm.iloc[0]) + ' (cm) at ' + str_round(df.px_per_cm_height.iloc[0],0) + ' pixels/cm'
        max_tile = str(min(df.height_cm.iloc[0],df.width_cm.iloc[0]))
        nearest_tile = str(5 * math.floor(float(max_tile)/5))
        medium = df.medium.iloc[0]
        result = str_round(utils_fastai.get_painting_confidence_from_folder(title,save_folder,model_folder_path = save_pre_path,binarize=False,one_vote=one_vote),round_to)
        # print(title,result)
        result_perc = str_round(100*float(result),round_to)
        # result_onevote = str(round(utils_fastai.get_painting_confidence_from_folder(title,save_folder,model_folder_path = save_pre_path,binarize=False,one_vote=one_vote),round_to))
        # result_onevote_perc = str(round(100*float(result_onevote),round_to))

        slice_sizes = util.get_slice_sizes(results)
        # print(slice_sizes)
        min_tile = str(slice_sizes[0])
        next_min_tile = str(int(int(slice_sizes[0])+5))
        #get closest pollock
        closest_df = utils_fastai.get_painting_closest_PMF(learner_results,float(result),one_vote=one_vote, binarize=False,catagory = pollock_categories)
        closest = closest_df.painting.iloc[0]
        closestTitle = learner_master[learner_master.file == closest].title.iloc[0]
        # closest = closest.split('(')[0]
        closest_val = str_round(closest_df.pollock_prob.iloc[0],round_to)
        # closest_diff = str(round(np.abs(closest_df.pollock_prob.iloc[0] - float(result)),round_to))
        closest_diff = ("{0:."+str(round_to)+"f}").format(np.abs(closest_df.pollock_prob.iloc[0] - float(result)))
        
        #get closest_im
        closest_df_im = utils_fastai.get_painting_closest_PMF(learner_results,float(result),one_vote=one_vote, binarize=False,catagory = np_categories)
        closest_im = closest_df_im.painting.iloc[0]
        closestTitle_im = learner_master[learner_master.file == closest_im].title.iloc[0]
        # closest = closest.split('(')[0]
        closest_val_im = str_round(closest_df_im.pollock_prob.iloc[0],round_to)
        # closest_diff_im = str(round(np.abs(closest_df_im.pollock_prob.iloc[0] - float(result)),round_to))
        closest_diff_im = ("{0:."+str(round_to)+"f}").format(np.abs(closest_df_im.pollock_prob.iloc[0] - float(result)))
        
        #get highest_im
        highest_df_im = utils_fastai.get_painting_closest_PMF(learner_results,1.0,one_vote=one_vote, binarize=False,catagory = np_categories)
        highest_im = highest_df_im.painting.iloc[0]
        highestTitle_im = learner_master[learner_master.file == highest_im].title.iloc[0]
        highest_val_im = str_round(highest_df_im.pollock_prob.iloc[0],round_to)
        # highest_diff_im = str(round(np.abs(highest_df_im.pollock_prob.iloc[0] - float(result)),round_to))
        
        #get lowest pollock
        lowest_df = utils_fastai.get_painting_closest_PMF(learner_results,0,one_vote=one_vote, binarize=False,catagory = pollock_categories)
        lowest = lowest_df.painting.iloc[0]
        lowestTitle = learner_master[learner_master.file == lowest].title.iloc[0]
        lowest_val = str_round(lowest_df.pollock_prob.iloc[0],round_to)    
        


        below,above = utils_fastai.get_above_below_PMF_percentage(learner_results,float(result),one_vote=one_vote, binarize=False,catagory = pollock_categories,round_to = round_to)
        # closest_ov_df = utils_fastai.get_painting_closest_PMF(learner_results,float(result),one_vote=one_vote, binarize=False,catagory = ('P','J'))
        # closest_ov = closest_ov_df.painting.iloc[0].split('(')[0]
        # closest_ov_val = str(round(closest_ov_df.pollock_prob.iloc[0],round_to))
        # closest_ov_diff = str(round(np.abs(closest_ov_df.pollock_prob.iloc[0] - float(result_onevote)),round_to))

        if one_vote:
            machine_confidence =learner_summary.painting_onevote_accuracy_mean.iloc[0]
        else:
            machine_confidence =learner_summary.painting_accuracy_mean.iloc[0]
        machine_confidence = ("{0:."+str(round_to)+"f}").format(100*machine_confidence)
        
        machine_accuracy_paintings = rubric.get_machine_accuracy(learner_results,learner_master,remove_special = remove_special,catagory = pollock_categories,one_vote = one_vote,use_images = False,percent = round_to,binarize = False,binarize_prob=0.5,decision_prob=classification_thresh)
        machine_accuracy_paintings = str_round(machine_accuracy_paintings,round_to_acc)
        machine_accuracy_images = rubric.get_machine_accuracy(learner_results,learner_master,remove_special = remove_special,catagory = pollock_categories,one_vote = one_vote,use_images = True,percent = round_to,binarize = False,binarize_prob=0.5,decision_prob=classification_thresh)
        machine_accuracy_images = str_round(machine_accuracy_images,round_to_acc)

        #get similar sized painting

        valid_set = learner_metrics.valid_set.apply(eval).iloc[0]
        sim_pollock = utils_fastai.get_similar_min_dim_sized_painting(title,master,learner_master[learner_master.file.isin(valid_set)],catagory = pollock_categories)
        
        select_r= util.vote_system(learner_results,one_vote=one_vote, binarize=False)
        #get select painting in learner results
        if not select_painting:
            select_painting = sim_pollock
        
        select_df = select_r[select_r.painting == select_painting]
        select = select_df.painting.iloc[0]
        selectTitle = learner_master[learner_master.file == select].title.iloc[0]
        select_val = str_round(select_df.pollock_prob.iloc[0],round_to)
        select_diff= str_round(np.abs(select_df.pollock_prob.iloc[0] - float(result)),round_to)

                #get select painting in learner results
        if not select_painting_2:
            select_painting_2 = sim_pollock
        select_df_2 = select_r[select_r.painting == select_painting_2]
        select_2 = select_df_2.painting.iloc[0]
        selectTitle_2 = learner_master[learner_master.file == select_2].title.iloc[0]
        select_val_2 = str_round(select_df_2.pollock_prob.iloc[0],round_to)
        select_diff_2= str_round(np.abs(select_df_2.pollock_prob.iloc[0] - float(result)),round_to)
        
        
        sim_P_result = str_round(utils_fastai.get_painting_confidence_from_folder(sim_pollock,learner_folder,model_folder_path = container,binarize=False,one_vote=one_vote),round_to)
        # sim_P_result_onevote = str(round(utils_fastai.get_painting_confidence_from_folder(sim_pollock,learner_folder,model_folder_path = container,binarize=False,one_vote=one_vote),round_to))
        # print(sim_P_result)
        sim_P_title = learner_master[learner_master.file == sim_pollock].title.iloc[0]
        
        #num slices in blue poles
        bluepole_slices = util.get_num_slices_in_painiting(learner_master,'P108(J)',folders)
        
        #this painting num
        total_slices = util.get_num_slices_in_painiting(master,title,folders)
        
        classification = str(True if float(result) > classification_thresh else False)
        in_sentence_classification = 'does' if float(result) > classification_thresh else 'does NOT'
        
        #get confidence
        confidence = 'XXX'

        #get_special_stats
        print('get special stats')
        #I think get_folder_data takes some time. Ship to skip since redundant?
        r_AI,results_AI = stats.get_folder_data('AI_standard',load_results = True,container_folder = 'painting_preds',notes = '',img_path = 'Imitations_Summer_2023/Processed/Raw/')

        # AI_len = str(round(len(r_AI),round_to))
        result_list = [value.rsplit('_', 1)[0] for value in r_AI.painting]
        AI_len =str(len(set(result_list)))
        AI_paintings_accuracy = str_round(100*(1-len(r_AI[r_AI.pollock_prob >classification_thresh])/len(r_AI)),round_to)
        #I think get_folder_data takes some time
        r_T2,results_T2 = stats.get_folder_data('T2_standard',load_results = True,container_folder = 'painting_preds',notes = '',img_path = 'Test_Imitations-2/Processed/Raw')
        test_paintings_accuracy = str_round(100*(1-len(r_T2[r_T2.failed])/len(r_T2)),round_to)
        r_BW,results_BW = stats.get_folder_data('bw_standard',load_results = True,container_folder = 'painting_preds',notes = '',img_path = 'BW_converted/Processed/Raw/')
        # print('bw_compare')
        BW_C_compare = util.compare_BW_converted(full_results,results_BW) #changed to not include remove special
        # print('finish bw compare')
        BWC_abs_diff = str_round(BW_C_compare.abs_PMF_diff.mean(),round_to)
        BWC_PMF_diff = str_round(BW_C_compare.PMF_diff.mean(),round_to)
        BW_PMF = str_round(BW_C_compare.pollock_prob_BW.mean(),round_to)
        C_PMF = str_round(BW_C_compare.pollock_prob_C.mean(),round_to)

        r_Henri,results_Henri = stats.get_folder_data('henri_standard',load_results = True,container_folder = 'painting_preds',notes = '',img_path = 'Henri/Processed/Raw')
        Henri_mean = str_round(r_Henri.pollock_prob.mean(),round_to)
        Henri_len = str_round(len(r_Henri),round_to)
        Henri_failed = str_round(len(r_Henri[r_Henri.failed]),round_to)
        Henri_accuracy = str_round(100*(1-float(Henri_failed)/float(Henri_len)),round_to)

        pollock_mean = str_round(full_r[full_r.painting.str.startswith(pollock_categories)].pollock_prob.mean(),round_to)
        non_pollock_mean = str_round(full_r[~full_r.painting.str.startswith(pollock_categories)].pollock_prob.mean(),round_to)
        P_mean = str_round(full_r[full_r.painting.str.startswith('P')].pollock_prob.mean(),round_to)
        J_mean = str_round(full_r[full_r.painting.str.startswith('J')].pollock_prob.mean(),round_to)
        PJ_diff = str_round(full_r[full_r.painting.str.startswith('J')].pollock_prob.mean()-full_r[full_r.painting.str.startswith('P')].pollock_prob.mean(),round_to)

        P_BW_diff = str_round(full_r[full_r.painting.str.startswith('P')].pollock_prob.mean()-r_BW.pollock_prob.mean(),round_to)

        if dummy:
            title_testing = dummy

        # dial_title_list = [str(learner_master[learner_master.file == name].title.iloc[0]) for name in dial_paintings]
        dial_title_list =['\\textit{' + str(learner_master[learner_master.file == name].title.iloc[0]) + '} (' + str(learner_master[learner_master.file == name].artist.iloc[0]) + ')' for name in dial_paintings]
        dial_titles = ', '.join(dial_title_list[:-1]) + ', and ' + dial_title_list[-1]

        PJ = pd.read_csv(os.path.join(container,learner_folder,stats_sub_dir,'stats.csv'), index_col=0)
        select_SI = str_round(PJ[PJ.painting == select].variation.iloc[0],round_to_acc)
        select_M = str_round(PJ[PJ.painting == select].mag.iloc[0],round_to_acc)
        select_U = str_round(PJ[PJ.painting == select].signal_uniformity.iloc[0],round_to_acc)
        select_C = str_round(PJ[PJ.painting == select].area_coverage.iloc[0],round_to_acc)
        select_year = str(learner_master[learner_master.file == select].year.dt.year.iloc[0])

        select_2_SI = str_round(PJ[PJ.painting == select_2].variation.iloc[0],round_to_acc)
        select_2_M = str_round(PJ[PJ.painting == select_2].mag.iloc[0],round_to_acc)
        select_2_U = str_round(PJ[PJ.painting == select_2].signal_uniformity.iloc[0],round_to_acc)
        select_2_C = str_round(PJ[PJ.painting == select_2].area_coverage.iloc[0],round_to_acc)
        select_2_year = str(learner_master[learner_master.file == select_2].year.dt.year.iloc[0])


        #Start plotting
        
        cropped_path = Path(img_path_testing,'Full')
        if not os.path.exists(cropped_path):
            cropped_path = Path(cropped_path.parents[1],'Raw','Full') 
        cropped_painting = [item for item in os.listdir(cropped_path) if  item.startswith(title + '_c')]
        assert len(cropped_painting) >= 1, 'check that the painting is in the Full folder'
        cropped_painting = cropped_painting[0]
        # print(cropped_painting)
        full_img_path = Path(cropped_path,cropped_painting)
        # Set a new value for the maximum image pixels
        new_max_image_pixels = 1000000000  # Set your desired value here

        # Update the MAX_IMAGE_PIXELS constant
        Image.MAX_IMAGE_PIXELS = new_max_image_pixels
        img =Image.open(full_img_path)
        img = ImageOps.contain(img, (2048,2048))
        img.save(Path(save_pre_path,save_folder,sub_dir,'analyzed_image.png'))
        
        #slice grids
        plotting.plot_slice_grid(name = title,folder = img_path_testing ,slice_size = '25',figsize=20,axes_pad=0.3,save=Path(save_pre_path,save_folder,sub_dir,'25_grid.png'),dpi=10)

        #first for the test painting
        plotting.make_row_hist_combined(title,img_path_testing,results,
                                        save_pre_path=save_pre_path,
                                        hist_save_name='hist.png',
                                        save_folder=os.path.join(save_folder,sub_dir),
                                        one_vote = one_vote)
        

        #select
        plotting.make_row_hist_combined(select,img_path,learner_results,
                                        save_pre_path=save_pre_path,
                                        hist_save_name='hist_select.png',save_folder=os.path.join(save_folder,sub_dir),one_vote = one_vote)

        
        
    ####Make dial

        
        # ext= cropped_painting.split(title)[1] #'_cropped.tif'
        # print(ext)

        test_img_prob = float(result)
        # print(test_img_prob)
        if os.path.exists('Paintings/Processed/Raw/Full/'):
            path_to_images='Paintings/Processed/Raw/Full/'
            ext = '_cropped.tif'
        elif os.path.exists('Paintings/Processed/Descreened/Full/'):
            path_to_images='Paintings/Processed/Descreened/Full/'
            ext='_cropped_descreened.tif'
        else:
            print('test image path does not exist')


        plotting.make_pollock_dial(title,result,
                            test_image_path = img_path_testing,
                            learner_folder=learner_folder,
                            test_ext = test_ext,
                            lines = True,
                            #   dial_paintings = ['F34','D15', 'C2','A12', 'A66', 'P2(V)', 'P4(F)', 'C63','black', 'F65','P68(V)'],
                            dial_paintings = dial_paintings, # 'F102''P86(S)','F34'
                            remove_special = remove_special,
                            save_name = 'dial.png',
                            show_fig = True,
                            specify_results = False, #results
                            save_pre_path = save_pre_path,
                            save_folder = os.path.join(save_folder,sub_dir),
                            round_to = round_to,
                            threshold = classification_thresh
                            )




        scales = pd.read_csv(os.path.join(container,learner_folder,stats_sub_dir,'scales.csv'), index_col=0)
        PJ = pd.read_csv(os.path.join(container,learner_folder,stats_sub_dir,'stats.csv'), index_col=0)
        NP = pd.read_csv(os.path.join(container,learner_folder,stats_sub_dir,'stats_NP.csv'), index_col=0)


        stats.plot_scales(scales,width_factor=1,selected_paths = False,save = os.path.join(save_pre_path,save_folder,sub_dir,'average_zoom_ins.png'))
        # PJ = stats.get_stats(full_results,scales,learner_master,categories = ('P','J'),verbose =True)
        print('get test painting stats')
        test_scales = stats.get_scales(results,categories=False,verbose = False,save =False,add_max = True)
        
        test_stats = stats.get_stats(results,test_scales,master,categories = False,verbose =True, 
                        viz_path_start = os.path.join('viz',sub_folder,folder_str),
                        viz_path_end =  folder_str.split(title+'_')[1],
                        do_year = False
                        )
        stats.plot_stats(PJ[PJ.painting.isin(remove_special)],
                         test_stats_or_list = test_stats[test_stats.painting == title],
                         save = os.path.join(save_pre_path,save_folder,sub_dir,'timeline.png'))
        

        r_res= pd.read_csv(os.path.join(container,learner_folder,stats_sub_dir,'r_res.csv'))
        # results_res =pd.read_csv(os.path.join(container,learner_folder,stats_sub_dir,'results_res.csv'))

        res_path = 'Paintings/Processed/Res'
        res_painting = 'P108(L Left)'
        special_res_path = os.path.join(res_path,str(res_tile_cm), res_painting + '_1_divided_by_'+str(res_factor) + '_cropped_C00_R00.tif')
        if not os.path.exists(special_res_path):
            print('making res tile for ' + res_painting)
            master_res = pd.read_parquet('master_res.parquet')
            util.tile_res_specific(res_painting + '_1_divided_by_'+str(res_factor),master_res,
                    output_size_cm = res_tile_cm,
                    res_factor = res_factor,
                    img_path_testing = res_path,
                    verbose = False,
                    center_tile = True,
                    add = 0,
                    update = False
                    )

        testing_res_path = os.path.join(Path(img_path_testing).parent,'Res',str(res_tile_cm))
        res_files = [file for file in util.os_listdir(testing_res_path) if file.startswith(title + '_1_divided_by_'+str(res_factor))]
        orig_files = [file for file in util.os_listdir(os.path.join(img_path_testing,str(res_tile_cm))) if file.startswith(title + '_cropped')]
        paths = [[os.path.join(img_path_testing,str(res_tile_cm),orig_files[0]),os.path.join(img_path,str(res_tile_cm),res_painting + '_cropped_C00_R00.tif')],
                [os.path.join(testing_res_path,res_files[0]), special_res_path]]
        stats.plot_comparison_res(r_res,include_1 = full_r,legend = False,save = os.path.join(save_pre_path,save_folder,sub_dir,'res.png'),selected_paths =paths,show_fig = True)

        # slice_sizes = ['RF=1, 20cm','RF=20, 20cm','RF=1, 20cm','RF=20, 20cm']  #util.get_slice_sizes(rresults)
        # res_row_save = os.path.join(save_pre_path,save_folder,sub_dir,'res_compare.png')
        # paths = ['Paintings/Processed/Raw/20/P71(L)_cropped_C00_R00.tif',
        #         'resolution/Processed/Raw/20/P71(L)_1_divided_by_20_cropped_C00_R00.tif',
        #         'Paintings/Processed/Raw/20/C14_cropped_C0_R0.tif',
        #         'resolution/Processed/Raw/20/C14_1_divided_by_20_cropped_C0_R0.tif'] #Would be good to have it automatically run the tiler for the painting in question
        # plotting.make_slice_row_fig(name = title,
        #                             slice_sizes=slice_sizes,
        #                             folder = img_path_testing,
        #                             figsize=(12, 4), 
        #                             dpi=80,
        #                             selected_thumbnails = False,
        #                             selected_paths =paths,
        #                             save = res_row_save)

        #Clean up some images with extra white space around the edges
        files_to_clean = ['timeline.png',
                          'average_zoom_ins.png',
                          'select_0_overlay.png',
                          'select_1_overlay.png',
                          'select_2_overlay.png',
                          'select_0_pollock_map.png',
                          'select_1_pollock_map.png',
                          'select_2_pollock_map.png',
                          'res.png']
        buffers = [0,
                   2,
                   0,
                   0,
                   0,
                   1,
                   1,
                   1,
                   2]
        for i,file in enumerate(files_to_clean):
            print(file)
            img = plotting.remove_white_space(cv2.imread(os.path.join(save_pre_path,save_folder,sub_dir,file)), threshold = 255,buffer = buffers[i])
            cv2.imwrite(os.path.join(save_pre_path,save_folder,sub_dir,file),img)

        PJ = pd.merge(left = PJ,right = learner_master[['file','set']],on = 'file')
        NP = pd.merge(left = NP,right = learner_master[['file','set']],on = 'file')
        SI_P = str_round(PJ.variation.mean(),round_to_acc)
        U_P = str_round(PJ.signal_uniformity.mean(),round_to_acc)
        C_P = str_round(PJ.area_coverage.mean(),round_to_acc)
        PMF_P = str_round(PJ.PMF.mean(),round_to)
        selection_logic_PJ = (PJ.set != 'train') & (~PJ.file.isin(tuple(remove_special)))
        selection_logic_NP = (NP.set != 'train') & (~NP.file.isin(tuple(remove_special)))
        SI_P_Test = str_round(PJ[selection_logic_PJ].variation.mean(),round_to_acc)
        U_P_Test = str_round(PJ[selection_logic_PJ].signal_uniformity.mean(),round_to_acc)
        C_P_Test = str_round(PJ[selection_logic_PJ].area_coverage.mean(),round_to_acc)
        PMF_P_Test = str_round(PJ[selection_logic_PJ].PMF.mean(),round_to)
        SI_NP = str_round(NP.variation.mean(),round_to_acc)
        U_NP = str_round(NP.signal_uniformity.mean(),round_to_acc)
        C_NP = str_round(NP.area_coverage.mean(),round_to_acc)
        PMF_NP = str_round(NP.PMF.mean(),round_to)
        SI_NP_Test = str_round(NP[selection_logic_NP].variation.mean(),round_to_acc)
        U_NP_Test = str_round(NP[selection_logic_NP].signal_uniformity.mean(),round_to_acc)
        C_NP_Test = str_round(NP[selection_logic_NP].area_coverage.mean(),round_to_acc)
        PMF_NP_Test = str_round(NP[selection_logic_NP].PMF.mean(),round_to)
        SI = str_round(test_stats[test_stats.painting == title].variation.iloc[0],round_to_acc)
        M = str_round(test_stats[test_stats.painting == title].mag.iloc[0],round_to_acc)
        U = str_round(test_stats[test_stats.painting == title].signal_uniformity.iloc[0],round_to_acc)
        C = str_round(test_stats[test_stats.painting == title].area_coverage.iloc[0],round_to_acc)

        P_set = set(learner_master[learner_master.group == 'P'].catalog)
        J_set = set(learner_master[learner_master.group == 'J'].catalog)
        len_non_dup = len(P_set.symmetric_difference(J_set))
        len_dup = len(P_set.intersection(J_set))
        P_list = list(learner_master[learner_master.group == 'P'].catalog)
        J_list = list(learner_master[learner_master.group == 'J'].catalog)
        combined_list = P_list + J_list
        result_list = [item for item in set(combined_list) if combined_list.count(item) > 1]
        len_multiple_images = str(len(result_list))
        counts = pd.read_csv(Path(Path(img_path).parts[0],'counts.csv'))
        PJ_tile_count = "{:,}".format(counts[counts.artist =='Pollock'].expected.sum())
        NP_tile_count = "{:,}".format(counts[counts.file.str.startswith(np_categories)].expected.sum())
        classification_thresh = str_round(classification_thresh, round_to)
        blue_poles_count = str(counts.groupby('file').in_folder.sum()['P108(J)'])

        if float(result) >= float(classification_thresh):
            smaller_larger = 'larger'
        else:
            smaller_larger = 'smaller'

        orientation_diff = str_round(pd.concat(r).mean().pollock_prob - float(result),round_to)
        orientation_diff = orientation_diff if orientation_diff[0] == '-' else '+' + orientation_diff
        NP_orientation_diff = str_round(pd.concat(rr)[pd.concat(rr).is_pollock == 'False']['diff'].mean(),round_to)
        NP_orientation_diff = NP_orientation_diff if NP_orientation_diff[0] == '-' else '+' + NP_orientation_diff
        PJ_orientation_diff = str_round(pd.concat(rr)[pd.concat(rr).is_pollock == 'True']['diff'].mean(),round_to)
        PJ_orientation_diff = PJ_orientation_diff if PJ_orientation_diff[0] == '-' else '+' + PJ_orientation_diff

        #make report data
        d= {'title':[title_testing],
            'submitted':[submitted_by],
            'canvas':[canvas_size],
            'medium':[medium],
            'result':[result],
            'resultP':[result_perc],
            'classThresh':[classification_thresh],
            'classification':[classification],
            'classStr':[in_sentence_classification],
            'confidence':[confidence],
            # 'onevote':[result_onevote],
            # 'onevoteP':[result_onevote_perc],
            'closest':[closestTitle],
            'closestval':[closest_val],
            'closestdff':[closest_diff],
            'select':[selectTitle],
            'selectval':[select_val],
            'selectdff':[select_diff],
            'select_SI':[select_SI],
            'select_M':[select_M],
            'select_U':[select_U],
            'select_C':[select_C],
            'select_year':[select_year],
            'select_2':[selectTitle_2],
            'selectval_2':[select_val_2],
            'selectdff_2':[select_diff_2],
            'select_2_SI':[select_2_SI],
            'select_2_M':[select_2_M],
            'select_2_U':[select_2_U],
            'select_2_C':[select_2_C],
            'select_2_year':[select_2_year],
            'lowestP':[lowestTitle],
            'lowestPval':[lowest_val],
            'highestI':[highestTitle_im],
            'highestIval':[highest_val_im],
            'closestdff':[closest_diff],
            'closestim':[closestTitle_im],
            'closestvalim':[closest_val_im],
            'closestdffim':[closest_diff_im],
            'aboveP':[str_round(above,0)],
            'belowP':[str_round(below,0)],
            # 'closestOV':[closest_ov],
            # 'closestOVval':[closest_ov_val],
            # 'closestOVdff':[closest_ov_diff],
            'MC':[machine_confidence],
            # 'MConevote':[machine_confidence_onevote],
            'MApaintings':[machine_accuracy_paintings],
            'MAimages':[machine_accuracy_images],
            'maxtile':[max_tile],
            'nearesttile':[nearest_tile],
            'mintile':[min_tile],
            'nextmintile':[next_min_tile],
            'simPtitle':[sim_P_title],
            'simPresult':[sim_P_result],
            'numslicesBluePoles':[bluepole_slices],
            'numslicesThis':[total_slices],
            'AI_len':[AI_len],
            'AI_accuracy':[AI_paintings_accuracy],
            'test_accuracy':[test_paintings_accuracy],
            'BWC_abs_diff' :[BWC_abs_diff],
            'BWC_PMF_diff' : [BWC_PMF_diff],
            'BW_PMF' : [BW_PMF],
            'C_PMF' : [C_PMF],
            'Henri_mean':[Henri_mean],
            'Henri_len':[Henri_len],
            'Henri_failed':[Henri_failed],
            'Henri_accuracy':[Henri_accuracy],
            'P_mean':[P_mean], #P just stands for P set in this case
            'J_mean':[J_mean],
            'PJ_diff':[PJ_diff],
            'P_BW_diff':[P_BW_diff],
            'dial_titles':[dial_titles],
            'SI_P':[SI_P], #P stands for all pollocks in this case and following
            'U_P':[U_P],
            'C_P':[C_P],
            'PMF_P': [PMF_P],
            'SI_P_Test':[SI_P_Test],
            'U_P_Test':[U_P_Test],
            'C_P_Test':[C_P_Test],
            'PMF_P_Test': [PMF_P_Test],
            'SI_NP':[SI_NP],#NP stands for all non-pollocks in this case and following
            'U_NP':[U_NP],
            'C_NP':[C_NP],
            'PMF_NP': [PMF_NP],
            'SI_NP_Test':[SI_NP_Test],
            'U_NP_Test':[U_NP_Test],
            'C_NP_Test':[C_NP_Test],
            'PMF_NP_Test': [PMF_NP_Test],
            'SI':[SI],
            'M':[M],
            'U':[U],
            'C':[C],
            'pollock_mean':[pollock_mean],
            'non_pollock_mean':[non_pollock_mean],
            'len_non_dup':[len_non_dup], #Really this means in P but not J or vice versa
            'len_dup ':[len_dup],
            'len_multiple_images':[len_multiple_images],
            'PJ_tile_count':[PJ_tile_count],
            'NP_tile_count':[NP_tile_count],
            'blue_poles_count':[blue_poles_count],
            'smaller_larger':[smaller_larger],
            groups[0]:[str_round(r[groups[0]].pollock_prob.iloc[0],round_to)],
            groups[1]:[str_round(r[groups[1]].pollock_prob.iloc[0],round_to)],
            groups[2]:[str_round(r[groups[2]].pollock_prob.iloc[0],round_to)],
            groups[3]:[str_round(r[groups[3]].pollock_prob.iloc[0],round_to)],
            groups[4]:[str_round(r[groups[4]].pollock_prob.iloc[0],round_to)],
            'orientation_mean':[str_round(pd.concat(r).mean().pollock_prob,round_to)],
            'orientation_diff':[orientation_diff],
            'PJ_' + groups[0]:[str_round(rr[groups[0]][rr[groups[0]].is_pollock == 'True']['pollock_prob_Aug'].iloc[0],round_to)],
            'PJ_' + groups[1]:[str_round(rr[groups[1]][rr[groups[1]].is_pollock == 'True']['pollock_prob_Aug'].iloc[0],round_to)],
            'PJ_' + groups[2]:[str_round(rr[groups[2]][rr[groups[2]].is_pollock == 'True']['pollock_prob_Aug'].iloc[0],round_to)],
            'PJ_' + groups[3]:[str_round(rr[groups[3]][rr[groups[3]].is_pollock == 'True']['pollock_prob_Aug'].iloc[0],round_to)],
            'PJ_' + groups[4]:[str_round(rr[groups[4]][rr[groups[4]].is_pollock == 'True']['pollock_prob_Aug'].iloc[0],round_to)],
            'PJ_Orig':[str_round(rr[groups[4]][rr[groups[4]].is_pollock == 'True']['pollock_prob_Orig'].iloc[0],round_to)],
            'PJ_orientation_mean': [str_round(pd.concat(rr)[pd.concat(rr).is_pollock == 'True'].pollock_prob_Aug.mean(),round_to)],
            'PJ_orientation_diff': [PJ_orientation_diff],
            'NP_' + groups[0]:[str_round(rr[groups[0]][rr[groups[0]].is_pollock == 'False']['pollock_prob_Aug'].iloc[0],round_to)],
            'NP_' + groups[1]:[str_round(rr[groups[1]][rr[groups[1]].is_pollock == 'False']['pollock_prob_Aug'].iloc[0],round_to)],
            'NP_' + groups[2]:[str_round(rr[groups[2]][rr[groups[2]].is_pollock == 'False']['pollock_prob_Aug'].iloc[0],round_to)],
            'NP_' + groups[3]:[str_round(rr[groups[3]][rr[groups[3]].is_pollock == 'False']['pollock_prob_Aug'].iloc[0],round_to)],
            'NP_' + groups[4]:[str_round(rr[groups[4]][rr[groups[4]].is_pollock == 'False']['pollock_prob_Aug'].iloc[0],round_to)],  
            'NP_Orig':[str_round(rr[groups[4]][rr[groups[4]].is_pollock == 'False']['pollock_prob_Orig'].iloc[0],round_to)],     
            'NP_orientation_mean': [str_round(pd.concat(rr)[pd.concat(rr).is_pollock == 'False'].pollock_prob_Aug.mean(),round_to)],    
            'NP_orientation_diff': [NP_orientation_diff],  
            # 'simPresultOV':[sim_P_result_onevote],
            }
        

        report_df = pd.DataFrame(data=d)
        temp_list =list(zip(report_df.columns.tolist(),report_df.iloc[0].tolist()))
        report_df = pd.DataFrame(data = temp_list,columns = ['variables','values'])
        # new_df.to_csv('test.csv',index=False)
        report_df.to_csv(Path(save_pre_path,save_folder,'report_data.csv'),index=False)
