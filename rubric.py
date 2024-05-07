import pandas as pd
import util
import os
from pathlib import Path

from fastai.vision.all import *
from fastai.data.all import *


def get_results_summary(folder):
    results= pd.read_csv(Path(folder,'results.csv'), low_memory=False)
    r = util.vote_system(results,one_vote=False, binarize=False, binarize_prob=0.5, decision_prob=0.50)
    r2 = util.vote_system(results,one_vote=True, binarize=False, binarize_prob=0.5, decision_prob=0.50)
    r = util.get_accuracy_column(r,sort = 'accuracy')
    r2 = util.get_accuracy_column(r2,sort = 'accuracy')
    results = util.get_accuracy_column(results,sort = 'accuracy')

    painting_accuracy_mean = r.accuracy.mean()
    painting_accuracy_std = r.accuracy.std()
    less_than_half_paintings = r[r.accuracy <0.5].painting.tolist()
    less_than_half_accuracies =r[r.accuracy <0.5].accuracy.tolist()
    less_than_half_accuracies = str([round(ii,2) for ii in less_than_half_accuracies])
    slice_accuracy_mean = results.accuracy.mean()
    slice_accuracy_std = results.accuracy.std()
    # print(results.head(2))
    PJ_diff = util.p_jpcr_diff(results,vote=False)
    PJ_diff_mean = PJ_diff['prob_diff'].loc['mean']
    PJ_diff_std = PJ_diff['prob_diff'].loc['std']
    painting_onevote_accuracy_mean = r2.accuracy.mean()
    painting_onevote_accuracy_std = r2.accuracy.std()
    PJ_diff_onevote = util.p_jpcr_diff(results,vote=True)
    PJ_diff_onevote_mean = PJ_diff_onevote['prob_diff'].loc['mean']
    PJ_diff_onevote_std = PJ_diff_onevote['prob_diff'].loc['std']
    less_than_half_paintings_onevote = str(r2[r2.accuracy <0.5].painting.tolist())
    less_than_half_accuracies_onevote = r2[r2.accuracy <0.5].accuracy.tolist()
    less_than_half_accuracies_onevote = str([round(ii,2) for ii in less_than_half_accuracies_onevote])
    paintings = str(sorted(r.painting.tolist()))
    
    metrics= pd.read_csv(Path(folder,'metrics.csv'), low_memory=False)
    description_str = get_description_str(metrics)
    d = {'folder':str(folder),
         'run_params':description_str,
         'num_paintings':len(r),
         'painting_accuracy_mean':painting_accuracy_mean,
         # 'painting_accuracy_std':painting_accuracy_std,
         'PJ_diff_mean':PJ_diff_mean,
         'PJ_diff_onevote_mean':PJ_diff_onevote_mean,
         # 'PJ_diff_std':PJ_diff_std,
         'slice_accuracy_mean':slice_accuracy_mean,
         # 'slice_accuracy_std':slice_accuracy_std,
         'less_than_half_paintings':less_than_half_paintings,
         'less_than_half_accuracies':less_than_half_accuracies,
         'painting_onevote_accuracy_mean':painting_onevote_accuracy_mean,
         # 'painting_onevote_accuracy_std':painting_onevote_accuracy_std,
         'less_than_half_paintings_onevote':less_than_half_paintings_onevote,
         'less_than_half_accuracies_onevote':less_than_half_accuracies_onevote,
         'paintings':paintings,
        }
    return d

def get_summary_df(run_folder, container = 'runs',preds_folder = 'preds',save='default'):
    #initialize df
    summary = {'folder':[],
               'run_params':[],
               'num_paintings':[], 
               'painting_accuracy_mean':[],
               'painting_onevote_accuracy_mean':[],
               # 'painting_accuracy_std':[],
               'PJ_diff_mean':[],
               'PJ_diff_onevote_mean':[],
               # 'PJ_diff_std':[],           
               'slice_accuracy_mean':[],
               # 'slice_accuracy_std':[],  
               'less_than_half_paintings':[],
               'less_than_half_accuracies':[],

               # 'painting_onevote_accuracy_std':[],
               'less_than_half_paintings_onevote':[],
               'less_than_half_accuracies_onevote':[],
               'paintings':[]
            }

    # get run level details
    d =  get_results_summary(Path(container,run_folder))
    for key in list(summary.keys()):
        summary[key].append(d[key])
    # get the rest of the preds
    if os.path.exists(os.path.join(container,run_folder,preds_folder)):
        for folder in next(os.walk(os.path.join(container,run_folder,preds_folder)))[1]:
            if not folder.startswith('.') and os.path.exists(Path(container,run_folder,preds_folder,folder,'results.csv')):
                d = get_results_summary(Path(container,run_folder,preds_folder,folder))
                for key in list(summary.keys()):
                    summary[key].append(d[key])
    summary_df = pd.DataFrame(data=summary)
    # summary_df['less_than_half_paintings']=summary_df['less_than_half_paintings'].apply(lambda x: str(x))
    # summary_df['less_than_half_accuracies']=summary_df['less_than_half_accuracies'].apply(lambda x: str([round(i,2) for i in x]))
    # summary_df['less_than_half_paintings']=summary_df['less_than_half_paintings'].apply(lambda x: ','.join([str(ii) for ii in x]))
    # summary_df['less_than_half_accuracies']=summary_df['less_than_half_accuracies'].apply(lambda x: ','.join(["{:.2f}".format(ii) for ii in x]))
    if save == 'default':
        summary_df.to_csv(Path(container,run_folder,'summary.csv'))
    elif save:
        summary_df.to_csv(save)
    return summary_df

def get_description_str(metrics):
    Descreened = metrics.Descreened.iloc[0]
    multipath =  True if metrics.multipath.iloc[0] else False
    min_folder = str(min([int(ii) for ii in metrics.folders.apply(eval).iloc[0] if ii.isnumeric()]))
    gray = metrics.grayscale.iloc[0]
    seed = metrics.seed.iloc[0]
    if str(seed) == 'nan':
        return '^ with adjustments'
    else:
        return 'S->'+str(seed) + ' Des->'+str(Descreened)[0]+' rP->'+str(multipath)[0]+' Gr->'+str(gray)[0]+' mF->'+min_folder
    
def get_machine_accuracy(learner_results,master,
                         remove_special = ['P69(V)','P43(W)','JPCR_01031','A9(right)','JPCR_01088','P13(L)','P19(L)','P25(L)','P32(W)','P33(W Lowerqual)','P47(F Lowres)','P65(L)','P75(V)','P77(J)','P80(J)','P86(S)','P105(J)','P106(V)','P115(F Lowres)','A14(Left)'],
                         catagory = ('J','P'),
                         one_vote = True,
                         use_images = False,
                         percent = False, 
                         binarize = False,
                         binarize_prob=0.5,
                         decision_prob=0.56): 
    r = util.vote_system(learner_results, one_vote=one_vote, binarize=False, binarize_prob=binarize_prob, decision_prob=decision_prob)
    paintings = r.painting.tolist()
    paintings = sorted(list(set(paintings)-set(remove_special)))
    if use_images:
        total = len(paintings)
        num_pollocks = len([painting for painting in paintings if painting.startswith(catagory)])
        num_imitations = total-num_pollocks
    else:
        num_pollocks = len(master[np.logical_and(master.file.isin(paintings),master.file.str.startswith(catagory))].catalog.unique())
        num_imitations = len([painting for painting in paintings if not painting.startswith(catagory)])
        total = num_pollocks+num_imitations
    num_wrong = len(r[r.failed])
    machine_accuracy = (total-num_wrong)/total
    if percent:
        machine_accuracy = round(100*machine_accuracy,percent)
    return machine_accuracy