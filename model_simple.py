from fastai.vision.all import *
from fastai.data.all import *
import pandas as pd
import numpy as np
import timm
import os
import datetime

def filter_file_path_list(file_paths,filter_away_items):
    #file paths should be 'fastcore.foundation.L'
    # filter_away_items can be a list or tuple of file names (as seen in master.file)
    filter_away_items = [item + '_' for item in filter_away_items]
    # print('filter away items',sorted(filter_away_items))
    return file_paths.filter(lambda o:not o.stem.startswith(tuple(filter_away_items)))

def get_hold_out_files(M,hold_out_catalog,hold_out_files):
    #Set hold out items
    M.loc[M['catalog'].isin(hold_out_catalog),'hold_out'] = True
    M.loc[~M['catalog'].isin(hold_out_catalog),'hold_out'] = False
    #get hold out items
    hold_out_files = list(set(M[M.hold_out].file.tolist() + hold_out_files))
    return hold_out_files

def append_str_today(string):
    today = datetime.datetime.now()
    date_time = today.strftime("_%m-%d-%Y")
    return string + date_time

def folder_management(save_str,pre_path='runs',append_date = True,overwrite = False):
    if append_date:
        save_str = Path(pre_path,append_str_today(save_str))
    else:
        save_str = Path(pre_path,save_str)
    if not os.path.exists(pre_path):
        os.mkdir(pre_path)
    if os.path.exists(save_str):
        if overwrite:
            print('Warning: Folder '+str(save_str)+ ' exists already. overwriting')
            os.makedirs(save_str)
        elif not os.path.exists(Path(save_str,'results.csv')):
            print('overwriting because no results.csv folder in director')
            os.makedirs(save_str, exist_ok = True)
        else:
            while os.path.exists(save_str) and os.path.exists(Path(save_str,'results.csv')):
                extender = 'x'
                save_str = Path(str(save_str) + extender)
                print('ERROR: run name already exists! adding an extension =' + extender + '=!')
            os.makedirs(save_str, exist_ok = True)
    else:
        os.makedirs(save_str)
    return save_str
        
    
def split_test_train(paint_list, pct):
    valid_list = random.sample(paint_list, int(pct*len(paint_list)))
    train_list = list(set(valid_list).symmetric_difference(set(paint_list)))
    return valid_list, train_list

    
def get_validation_list(M,valid_set,hold_out_catalog,hold_out_files,save = False, equal_pop=True,pollock_groups = ('P','J')):
    hold_out_files=get_hold_out_files(M,hold_out_catalog,hold_out_files)
    # print(hold_out_files)
    if isinstance(valid_set,list):
        valid = sorted(list(set(valid_set)-set(hold_out_files)))
        # print(valid)
    elif isinstance(valid_set,float):      
        test_pct = valid_set
        #get pollock catalog list
        # print('pollocks',[item for item in hold_out_files if item.startswith(('P','J'))])
        P = M[M.file.str.startswith(tuple(pollock_groups))].catalog.tolist()
        # P = M[M.file.str.startswith(('P','J'))].catalog.tolist()
        #filter out the hold out catalogs
        P_filtered = list(set(P).difference(set(hold_out_catalog)))
        # get unique catalog numbers
        P_uniq_cat = list(set(P_filtered))
        #find imitations valid items
        # valid_P_cat = random.sample(P_uniq_cat, int(test_pct*len(P_uniq_cat)))
        # train_P_cat = list(set(valid_P_cat).symmetric_difference(set(P_uniq_cat)))
        valid_P_cat, train_P_cat = split_test_train(P_uniq_cat, test_pct)
        
        #get the corresponding files for those catalog numbers
        valid_P = M[M.catalog.isin(valid_P_cat)].file.tolist()
        train_P = M[M.catalog.isin(train_P_cat)].file.tolist()
        #filter away any lingering duplicates
        valid_P = list(set(valid_P).difference(set(hold_out_files)))
        train_P = list(set(train_P).difference(set(hold_out_files)))
        #get imiation file list
        I = M[~M.file.str.startswith(('P','J','B'))].file.tolist()
        #filter out the hold out files (need to include the imitations)
        I_filtered = list(set(I).difference(set(hold_out_files)))
        
        if equal_pop:
            categories = ['A', 'C', 'D', 'E', 'F', 'G']
            valid_I = []
            train_I = []
            for cat in categories:
                filtered_list = [ii for ii in I_filtered if ii.startswith(cat)]
                valid_cat, train_cat = split_test_train(filtered_list, test_pct)
                valid_I += valid_cat
                train_I += train_cat
            
            assert np.abs(len(valid_I) / (len(valid_I) + len(train_I)) - test_pct) < 0.01, f'something weird happened with equal pop \n selection {len(valid_I) /(len(valid_I) + len(train_I))} and test_pct {test_pct}'
        else:
            #find imitations valid items
            valid_I, train_I = split_test_train(I_filtered, test_pct)
            
        #combine the list of valid items
        valid = valid_P + valid_I
        train = train_P + train_I
        
        
        
    else:
        print('ERROR: valid_set is neither a float or a list')
    if save:
        #save the list for later use 
        valid_list = pd.DataFrame(sorted(valid), columns=["valid_list"])
        valid_list.to_csv(Path(save,'valid_list'), index=False)
        # To Read later valid_list = pd.read_csv(Path('runs','valid_list'))
        # valid = valid_list.valid_list.tolist()
    return valid

def is_pollock(x):
    if isinstance(x,str):
        return x.startswith(('P','J'))
    return x.name.startswith(('P','J'))

# class AlbumentationsTransform(DisplayedTransform):
#     split_idx,order=0,2
#     def __init__(self, train_aug): store_attr()
    
#     def encodes(self, img: PILImage):
#         aug_img = self.train_aug(image=np.array(img))['image']
#         return PILImage.create(aug_img)
# def get_train_aug(): return A.Compose([
#             # A.ToGray(p=0.5),
#             A.RandomBrightnessContrast(p=0.5),
#             # A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.5, hue=0.05,p=0.5),
#             # A.HorizontalFlip(p=0.5),
#             # A.VerticalFlip(p=0.5),
#             # A.RandomRotate90(p=0.5),
#         ])

def get_dls(M,valid,path,folders,hold_out_files=[],px_size = 256,grayscale=False,flipped = True,multipath = False,bs = 64):

    def get_image_files_filtered(path):
        
        file_paths = get_image_files_set_folders(path) #gets all files in select folders
        if multipath:
            file_paths_no_artifacts = filter_file_path_list(file_paths,M[M.artifacts == 'True'].file.tolist())
            file_paths_descreened = get_image_files_set_folders(multipath)
            file_paths_descreened = filter_file_path_list(file_paths_descreened,M[M.artifacts != 'True'].file.tolist())
            file_paths = L(file_paths_no_artifacts,file_paths_descreened).concat()
        # print('first filter')
        file_paths = filter_file_path_list(file_paths,hold_out_files) #Get's rid of hold out items
        # print(file_paths)
        # file_paths = filter_file_path_list(file_paths,M[M.file.str.startswith('P')].file.tolist())#get's rid of P's so we're just training on J's
        # print('second filter')
        file_paths = filter_file_path_list(file_paths,M[M.file.str.startswith('B')].file.tolist())#get's rid of B's because they are unkown    
        #get rid of any stems that appear in folders but not in master 
        file_stems = list(set([item.stem.split('_cropped')[0] for item in file_paths]))
        file_stems = list(set(file_stems).difference(set(M.file.tolist())))
        # print('file_stems',sorted(file_stems))
        # print('third filter')
        file_paths = filter_file_path_list(file_paths,file_stems)
        # print('file paths',len(file_paths))
        
        return file_paths

    def get_image_files_set_folders(path):
        return get_image_files(path, folders = folders)

    def get_tfms(px_size,flipped = False):
        if flipped:
            tfms = [Resize(px_size),FlipItem(0.5)]
        else:
            tfms = Resize(px_size)
        return tfms    
        
    valid = [item + '_' for item in valid]
    splitter_by_painting = FuncSplitter(lambda o:o.stem.startswith(tuple(valid))) #Note that Path() is not required since 'o' is already a path object
    if grayscale:
        imblock = ImageBlock(cls=PILImageBW)
    else:
        imblock = ImageBlock
        
    paintblock = DataBlock((imblock, CategoryBlock),
                       get_items = get_image_files_filtered,#get_image_files_set_folders,
                       get_y     = is_pollock,
                       splitter  = splitter_by_painting,#RandomSplitter(), #splitter,
                    #    item_tfms = [Resize(px_size),AlbumentationsTransform(get_train_aug()) ])
                       item_tfms = get_tfms(px_size))

    dls = paintblock.dataloaders(path,bs = bs)
    return dls

def save_learner(dls,learn,save_path,with_opt=True):
    # # save pickle
    with open(Path(save_path,'dls.pickle'), 'wb') as f:
        pickle.dump(dls, f)    
    # exports the weights that can be loaded into our model we define before loading (just like we did before out first fine tune)
    if with_opt:
        learn.save('learn',with_opt = with_opt)
    else:
        learn.save('learn')
    # # Just exports the model. Doesn't have info from the datablock
    learn.export('export.pkl')

#model_run(M,'10-Max',folders = folders,valid_set = valid_list,notes= notes,seed = 42)
def model_run(master, save_str, 
              Descreened = False, 
              valid_set = 0.2,
              seed = 0,
              thread_num = False,
              computer = '',
              px_size = 256,
              grayscale = False,
              epochs = 1,
              model = resnet50,
              folders = ['10','100','105', '110', '115', '120', '125',
                         '130', '135', '140', '145', '15', '150', '155', '160',
                         '165', '170', '175', '180', '185', '190', '195', '20',
                         '200', '205', '210', '215', '220', '225', '230', '235',
                         '240', '245', '25', '250', '255', '260', '265', '270',
                         '275', '280', '285', '290', '295', '30', '300', '305',
                         '310', '315', '320', '325', '330', '335', '340', '345',
                         '35', '350', '355', '360', '40', '45', '5', '50',
                         '55', '60', '65', '70', '75', '80', '85', '90',
                         '95','Max'],
              hold_out_catalog= ['93', '165', '179', '207', '231', '801', '820', '367', '371', '380','?'],
              hold_out_files = ['A57','A66','A64','A54(R)','C20','C55','C47(R)','D19','D8','E10','F114','F61','F110','F63','F94','F87','F99','G24','G58','G94','G62','G1','G27'],
              hold_out_duplicates = ['P13(L)','P19(L)','P25(L)','P32(W)','P33(W Lowerqual)',
'P47(F Lowres)','P65(L)','P75(V)','P77(J)','P80(J)','P86(S)','P105(J)','P106(V)','P115(F Lowres)','A14(Left)'],
              remove_duplicates = True,
              notes = '',
              overwrite = False,
              pretrained=True,
              pollock_groups = ['P','J'], # Just for the purposes of setting dist_metrics and get_validation set Doesn't change ispollock()
              multipath = False #Path('Paintings/Processed/Descreened/')
             ):
    #hold_out_files is typically used for just imitations because we want to hold out the entire catalog # for pollocks
    if seed:
        set_seeds(seed=seed,thread_num = thread_num)
    t = time.time()
    #valid_set can be a list that specifies the exact paintings in the validation set or a percentage
    save_path = folder_management(save_str,overwrite=True)
    if isinstance(valid_set, pd.DataFrame):
        valid_set = valid_set.valid_list.tolist()
    if Descreened:
        path = Path('Paintings/Processed/Descreened/') #Works on docker or locally
    else:
        path = Path('Paintings/Processed/Raw/')
    if remove_duplicates:
        hold_out_files_full = hold_out_files + hold_out_duplicates
    else:
        hold_out_files_full = hold_out_files
    hold_out_files_full = get_hold_out_files(master,hold_out_catalog,hold_out_files_full)
    # print(hold_out_files_full)
    
    ratio_df = pd.DataFrame(index=['numSlices', 'file'], columns=['ratio'])
    ratio_df['ratio'] = 0.5
    if isinstance(valid_set,float):
        print('valid set is a float?')
        while (np.abs(ratio_df['ratio']) > 0.05).any():
            valid = get_validation_list(master,valid_set,hold_out_catalog,hold_out_files_full,save = False,pollock_groups = tuple(pollock_groups))
            valid_ratio = find_dist_of_pollocks_imitations(valid, master, TRAIN=False)
            train_ratio = find_dist_of_pollocks_imitations(valid, master, TRAIN=True)
            ratio_df['ratio'] = valid_ratio['ratio'] - train_ratio['ratio']
    else:
        valid = get_validation_list(master,valid_set,hold_out_catalog,hold_out_files_full,save = False,pollock_groups = tuple(pollock_groups))
        valid_ratio = find_dist_of_pollocks_imitations(valid, master, TRAIN=False)
        train_ratio = find_dist_of_pollocks_imitations(valid, master, TRAIN=True)
    print('valid set has: \n', valid_ratio.to_string())
    print('train set has: \n', train_ratio.to_string())
    
    # print(len(valid),sorted(valid))
    dls = get_dls(master,valid,path, folders,hold_out_files=hold_out_files_full,px_size=px_size,grayscale = grayscale,multipath=multipath)
    # print(set([item.stem[0] for item in dls.train.items])) 
    # print(set([item.stem[0] for item in dls.valid.items]))
    valid_set_dls = sorted(list(set([item.stem.split('_c')[0] for item in dls.valid.items])))
    assert len(valid) == len(valid_set_dls), 'valid set does not match dls.valid.items differences are:' + str(sorted(set(valid).symmetric_difference(set(valid_set_dls))))
    valid_list = pd.DataFrame(sorted(valid_set_dls), columns=["valid_list"])
    valid_list.to_csv(Path(save_path,'valid_list'), index=False)
    
    get_dist_metrics(dls,save_path=save_path,groups = pollock_groups)
    
    learn = vision_learner(dls, model, metrics=error_rate, pretrained=pretrained)
    learn.path = save_path
    learn.fine_tune(epochs)
    save_learner(dls,learn,save_path)
    preds = make_predictions(learn,save_path)
    elapsed = time.time() - t
    today = datetime.datetime.now()
    date_time = today.strftime("%m-%d-%Y")
    metrics = model_analysis(dls,preds,save_path=save_path,runtime = elapsed,date = date_time,folders=folders,Descreened = Descreened,notes = notes,seed = seed,valid_set = valid_set_dls,px_size = px_size,grayscale=grayscale,epochs = epochs,hold_out_catalog=hold_out_catalog,hold_out_files = hold_out_files,computer = computer,hold_out_duplicates = hold_out_duplicates,hold_out_files_full= hold_out_files_full,model = model,multipath = multipath)
    
    if seed:
        reset_seeds()
      
    # return dls
    return metrics

def model_run2(master, save_str, 
              Descreened = False, 
              valid_set = 0.2,
              seed = 0,
              thread_num = False,
              computer = '',
              px_size = 256,
              grayscale = False,
              epochs = 1,
              arch = resnet50,
              folders = ['10','100','105', '110', '115', '120', '125',
                         '130', '135', '140', '145', '15', '150', '155', '160',
                         '165', '170', '175', '180', '185', '190', '195', '20',
                         '200', '205', '210', '215', '220', '225', '230', '235',
                         '240', '245', '25', '250', '255', '260', '265', '270',
                         '275', '280', '285', '290', '295', '30', '300', '305',
                         '310', '315', '320', '325', '330', '335', '340', '345',
                         '35', '350', '355', '360', '40', '45', '5', '50',
                         '55', '60', '65', '70', '75', '80', '85', '90',
                         '95','Max'],
              hold_out_catalog= ['93', '165', '179', '207', '231', '801', '820', '367', '371', '380','?'],
              hold_out_files = ['A57','A66','A64','A54(R)','C20','C55','C47(R)','D19','D8','E10','F114','F61','F110','F63','F94','F87','F99','G24','G58','G94','G62','G1','G27'],
              hold_out_duplicates = ['P13(L)','P19(L)','P25(L)','P32(W)','P33(W Lowerqual)',
'P47(F Lowres)','P65(L)','P75(V)','P77(J)','P80(J)','P86(S)','P105(J)','P106(V)','P115(F Lowres)','A14(Left)'],
              remove_duplicates = True,
              notes = '',
              overwrite = False,
              pollock_groups = ['P','J'], # Just for the purposes of setting dist_metrics and get_validation set Doesn't change ispollock()
              multipath = False, #Path('Paintings/Processed/Descreened/')
              path = 'Paintings/Processed/Raw/',
              bs = 64,
              pretrained=True,
              add_ho_to_valid = False,
              type = ''
             ):
    #hold_out_files is typically used for just imitations because we want to hold out the entire catalog # for pollocks
    if seed:
        set_seeds(seed=seed,thread_num = thread_num)
    t = time.time()
    #valid_set can be a list that specifies the exact paintings in the validation set or a percentage
    save_path = folder_management(save_str,overwrite=False)
    if isinstance(valid_set, pd.DataFrame):
        valid_set = valid_set.valid_list.tolist()
    # if Descreened:
    #     path = Path('Paintings/Processed/Descreened/') #Works on docker or locally
    # else:
    #     path = Path('Paintings/Processed/Raw/')
    path = Path(path)
    if remove_duplicates:
        hold_out_files_full = hold_out_files + hold_out_duplicates
    else:
        hold_out_files_full = hold_out_files
    hold_out_files_full = get_hold_out_files(master,hold_out_catalog,hold_out_files_full)
    # print(hold_out_files_full)
    
    ratio_df = pd.DataFrame(index=['numSlices', 'file'], columns=['ratio'])
    ratio_df['ratio'] = 0.5
    if isinstance(valid_set,float):
        print('valid set is a float?')
        while (np.abs(ratio_df['ratio']) > 0.05).any():
            valid = get_validation_list(master,valid_set,hold_out_catalog,hold_out_files_full,save = False,pollock_groups = tuple(pollock_groups))
            valid_ratio = find_dist_of_pollocks_imitations(valid, master, TRAIN=False)
            train_ratio = find_dist_of_pollocks_imitations(valid, master, TRAIN=True)
            ratio_df['ratio'] = valid_ratio['ratio'] - train_ratio['ratio']
    else:
        valid = get_validation_list(master,valid_set,hold_out_catalog,hold_out_files_full,save = False,pollock_groups = tuple(pollock_groups))
        valid_ratio = find_dist_of_pollocks_imitations(valid, master, TRAIN=False)
        train_ratio = find_dist_of_pollocks_imitations(valid, master, TRAIN=True)
    print('valid set has: \n', valid_ratio.to_string())
    print('train set has: \n', train_ratio.to_string())
    
    # print(len(valid),sorted(valid))
    if add_ho_to_valid:
        valid = valid + hold_out_files_full
        hold_out_files_full = []
    dls = get_dls(master,valid,path, folders,hold_out_files=hold_out_files_full,px_size=px_size,grayscale = grayscale,multipath=multipath,bs = bs)
    # print(set([item.stem[0] for item in dls.train.items])) 
    # print(set([item.stem[0] for item in dls.valid.items]))
    # arch = timm.create_model(arch,pretrained = pretrained,num_classes=dls.c)


    valid_set_dls = sorted(list(set([item.stem.split('_c')[0] for item in dls.valid.items])))
    assert len(valid) == len(valid_set_dls), 'valid set does not match dls.valid.items differences are:' + str(sorted(set(valid).symmetric_difference(set(valid_set_dls))))
    valid_list = pd.DataFrame(sorted(valid_set_dls), columns=["valid_list"])
    valid_list.to_csv(Path(save_path,'valid_list'), index=False)
    
    get_dist_metrics(dls,save_path=save_path,groups = pollock_groups)
    
    learn = vision_learner(dls, arch, metrics=error_rate,pretrained=pretrained)
    # learn = Learner(dls,arch,metrics=error_rate)
    learn.path = save_path
    learn.fine_tune(epochs)
    save_learner(dls,learn,save_path)
    preds = make_predictions(learn,save_path)
    elapsed = time.time() - t
    today = datetime.datetime.now()
    date_time = today.strftime("%m-%d-%Y")
    metrics,learner_results = model_analysis(dls,preds,save_path=save_path,runtime = elapsed,date = date_time,folders=folders,Descreened = Descreened,notes = notes,seed = seed,valid_set = valid_set_dls,px_size = px_size,grayscale=grayscale,epochs = epochs,hold_out_catalog=hold_out_catalog,hold_out_files = hold_out_files,computer = computer,hold_out_duplicates = hold_out_duplicates,hold_out_files_full= hold_out_files_full,model = arch,multipath = multipath)
    
    if seed:
        reset_seeds()
      
    # return dls
    if add_ho_to_valid:
        import rubric
        from util import str_round
        round_to_acc = 2
        round_to = 2
        one_vote = True
        classification_thresh = 0.56
        remove_special = ['P69(V)','P43(W)','JPCR_01031','A9(right)','JPCR_01088','P13(L)','P19(L)','P25(L)','P32(W)','P33(W Lowerqual)','P47(F Lowres)','P65(L)','P75(V)','P77(J)','P80(J)','P86(S)','P105(J)','P106(V)','P115(F Lowres)','A14(Left)']
        machine_accuracy_images = rubric.get_machine_accuracy(learner_results,master,remove_special = remove_special,catagory = ('J','P'),one_vote = one_vote,use_images = True,percent = round_to,binarize = False,binarize_prob=0.5,decision_prob=classification_thresh)
        machine_accuracy_images = str_round(machine_accuracy_images,round_to_acc)
        print('Maching Accuracy Images = ' + machine_accuracy_images)
        if len(save_str.split('--')) >1:
            arch_name = save_str.split('--')[1]
        else:
            arch_name = str(arch)
        d = {'save_path': [str(metrics.save_path.iloc[0])], 
            'arch':[arch_name],
            'Type':[type],
            'MA_images':[machine_accuracy_images],
            'folders':[str(metrics.folders.iloc[0])],
            'incorrect':[str(metrics.incorrect.iloc[0])],
            'runtime':[str(metrics.runtime.iloc[0])],
            'pretrained':[pretrained],
            'batch_size':[bs],
            'notes':[notes],
            'location_ran':[computer],
            'seed':[int(seed) if len(str(seed))>0 else seed],
            'valid_set':[valid_set],
            'px_size':[px_size],
            'epochs':[epochs],
            'hold_out_files_full':[hold_out_files_full],
            'model':[arch],
            'img_path':[path]
            }
        quick_ref = pd.DataFrame(data=d)
        quick_ref.to_csv(os.path.join(save_path,'quick_ref.csv'))

    return metrics

def set_seeds(seed = 42, thread_num = False):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if thread_num:
        torch.set_num_threads(thread_num)
        
def reset_seeds():
    seed = int(datetime.datetime.now().timestamp())
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)    
    torch.set_num_threads(multiprocessing.cpu_count())
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False   
    

def make_predictions(learn,save_path,ds_idx =1):
    preds = learn.get_preds()
    with open(Path(save_path,'preds.pickle'), 'wb') as f:
        pickle.dump(preds, f)
    return preds 


def model_analysis(dls,preds,save_path = False,prob_cutoff = 0.56,runtime = 0,date = '',folders = [],Descreened = [], notes = '',write_name = 'metrics.csv',valid_set = '',seed = '',px_size = '',grayscale = '',epochs = '',hold_out_catalog = [],hold_out_files=[], hold_out_duplicates = [],hold_out_files_full= [],computer = '',failed_column = 'sizesF_binaryT_failed',model='',multipath = ''):
    # print('dls',dls)
    results = get_results_df(dls,preds,save_path)   
    failed_list= results.loc[(results.prediction != results.actual)]
    # print(failed_list)
    compiled_results = do_all_voting(results, binarize_prob=prob_cutoff, decision_prob=prob_cutoff,simplify=True,failed = False)
    failed_paintings = compiled_results[compiled_results[failed_column]].painting.tolist()
    group_accuracies = get_group_accuracies(compiled_results,failed_column = failed_column)
    painting_error_rate = len(compiled_results[compiled_results[failed_column]])/len(compiled_results)
    run_error_rate = len(failed_list)/len(results) #this
    if isinstance(runtime,(int,float)):
        runtime = str(datetime.timedelta(seconds=round(runtime)))
    d = {'save_path': [save_path], 'run_error_rate':[run_error_rate],'group_accuracies':[group_accuracies],'painting_error_rate':[painting_error_rate],'incorrect':[failed_paintings],'runtime':[runtime],'date':[date],'folders':[folders],'Descreened':[Descreened],'notes':[notes],'location_ran':[computer],'seed':[int(seed) if len(str(seed))>0 else seed],'valid_set':[valid_set],'px_size':[px_size],'grayscale':[grayscale],'epochs':[epochs],'hold_out_catalog':[hold_out_catalog],'hold_out_files':[hold_out_files],'hold_out_duplicates':[hold_out_duplicates],'hold_out_files_full':[hold_out_files_full],'model':[model],'multipath':[multipath]}
    metrics = pd.DataFrame(data=d)
    if save_path:
        results.to_csv(Path(save_path,'results.csv'))
        compiled_results.to_csv(Path(save_path,'full_paints_results.csv'))
        metrics.to_csv(Path(save_path,write_name))
    
    return metrics, results

def get_group_accuracies(compiled_results,failed_column = 'sizesF_binaryT_failed'):
    groups = list(set([item[0] for item in compiled_results.painting.tolist()]))
    group_accuracies = {group:find_group_percentage(compiled_results,group,column=failed_column) for group in groups}
    group_accuracies = {key:f"{group_accuracies[key]:.2f}" for key in sorted(group_accuracies.keys())} #This
    return group_accuracies
    
def find_group_percentage(compiled_results,group,column = 'sizesF_binaryT_failed'):
    # num_incorrect = len(compiled_results[compiled_results.painting.str.startswith(group) & (compiled_results['prediction'] != compiled_results['actual'])])
    if column in compiled_results:
        num_correct = len(compiled_results[compiled_results.painting.str.startswith(group) & (compiled_results[column]==False)])
    else:
        num_correct = len(compiled_results[compiled_results.painting.str.startswith(group) & (compiled_results['prediction'] == compiled_results['actual'])])
    total_num = len(compiled_results[compiled_results.painting.str.startswith(group)])
    return num_correct/total_num    
                       

def get_results_df(dls,preds,save_path = False,write_name = 'results.csv', TRAIN=False):
    if isinstance(dls,fastai.data.core.DataLoaders):
        paths = dls.valid.items
        paths_train = dls.train.items
    else:
        paths = dls
        paths_train = dls #TRAIN has no meaning if paths are explicitely set
    def return_prediction(num,prob=0.56):
        if num > prob:
            return 1
        else:
            return 0
    predictions_og = [return_prediction(num[1]) for num in preds[0].tolist()]
    probabilities_og = [round(num[1],2) for num in preds[0].tolist()]
    
    if TRAIN:
        slice_size_og = [item.parent.name for item in paths_train]
        files_og = [item.name for item in paths_train]
    else:
        slice_size_og = [item.parent.name for item in paths]
        files_og = [item.name for item in paths]
    
    # print('img_path_og',img_path_og)
    actuals_og = preds[1].tolist()
    results = pd.DataFrame(list(zip(files_og,slice_size_og,predictions_og,actuals_og,probabilities_og)),columns = ['file', 'slice_size','prediction', 'actual','pollock_prob'])
    
    results = add_failed_paintings_to_results(results)
    # get painting_name
    results = add_painting_file_name_and_group_to_results(results)
    
    # print('img_paths',results.img_path.tolist())
    if save_path:    
        results.to_csv(Path(save_path,write_name))
    return results



def get_assessment_df_dict(painting_names_dict,results,prob_cutoff=0.5):
    slices_percent_correct = []
    predictions = []
    actuals = []
    painting_names = list(painting_names_dict.keys())
    for painting_name in painting_names:
        num_slices = int(painting_names_dict[painting_name])
        painting_results = results[results['file'].str.startswith(painting_name + '_')]
        slices_number_correct = len(painting_results[painting_results['prediction'] == is_pollock(Path(painting_name))])
        slices_percent_correct.append(slices_number_correct/num_slices)
        if slices_number_correct/num_slices>prob_cutoff:
            predict = is_pollock(Path(painting_name))
        else:
            predict = not is_pollock(Path(painting_name))
            
        predictions.append(predict)
        actuals.append(is_pollock(Path(painting_name)))
        
    df = pd.DataFrame(list(zip(painting_names,slices_percent_correct,predictions,actuals)),columns=('painting','slice percetange correct','prediction','actual'))
    return df

def load_dls_learner(folder,container_folder = 'runs',arch = resnet34,with_opt=True):
    path = Path(container_folder,folder)
    
    with open(Path(path,'dls.pickle'),'rb') as f:
        dls = pickle.load(f)
    learn = vision_learner(dls,arch,metrics=error_rate)
    learn.path = path
    if str(arch).startswith('<function'):            
        learn = learn.load('learn')
    else:
        learn = learn.load('learn',with_opt=with_opt)
    return dls,learn

def get_combined_df(folders,container_folder = 'runs',csv = 'metrics.csv'):
    df = []
    for folder in folders:
        if os.path.exists(Path(container_folder,folder,csv)): #excludes unfinished folders that don't have metrics yet
            df.append(pd.read_csv(Path(container_folder,folder,csv)))
    dfs = pd.concat(df)
    return dfs

def bar_compare_column(dfs,names,column,metric,title = ''):
    values = []
    for df in dfs:
        # print(getattr(df,column))
        field = getattr(df,column)
        # print(field)
        values.append(getattr(field,metric)())
        plt.ylabel(column + '(' + metric + ')')
        plt.title(title)
    plt.bar(names,values)
    
    
def painting_acc_by_slice_size(results,prob = True):
    
    if 'painting_name' not in results.columns:
        results = add_painting_file_name_and_group_to_results(results)
        
    if 'failed' not in results.columns:
        results = add_failed_paintings_to_results(results)
    if prob:
        painting_acc_at_sizes = results.groupby(['painting_name', 'slice_size'])['pollock_prob'].mean()
        acc_by_group = results.groupby(['group', 'slice_size'])['pollock_prob'].mean()
        error_painting = results.groupby(['painting_name', 'slice_size'])['pollock_prob'].std()
        error_group= results.groupby(['group', 'slice_size'])['pollock_prob'].std()
    else:
        pred_failed = results.groupby(['painting_name', 'slice_size']).sum()
        total_slices_by_painting = results.groupby(['painting_name', 'slice_size'])['file'].count()
        painting_acc_at_sizes = 1 -  pred_failed['failed'] / total_slices_by_painting

        pred_failed_by_group = results.groupby(['group', 'slice_size']).sum()
        total_slices_by_group = results.groupby(['group', 'slice_size'])['file'].count()
        acc_by_group = 1 - pred_failed_by_group['failed'] / total_slices_by_group
        
    
    return acc_by_group, painting_acc_at_sizes,error_group,error_painting

def add_painting_file_name_and_group_to_results(results):
    results['painting_name'] = results['file'].str.split('_cropped')
    results['painting_name'] = results['painting_name'].str[0]
    
    results['group'] = results['painting_name'].str[0]
    
    return results


def add_failed_paintings_to_results(results):
    results['failed'] = results['prediction'] != results['actual']
    return results
    
def get_group_props(items):
    groups = [item.stem[0] for item in items]
    total_len = len(groups)
    return {item:groups.count(item)/total_len for item in sorted(set(groups))}

def get_grouped_prop(items,groups = ['P','J'],group_name = 'pollocks'):
    group_props = get_group_props(items)
    group_fraction = sum([group_props[group] for group in groups])
    return {group_name:group_fraction,'other':1-group_fraction}

def get_dist_metrics(dls,save_path=False,write_name = 'dist_metrics.csv',groups = ['P','J']):
    group_props_T = get_group_props(dls.train.items)
    pollock_prop_T = get_grouped_prop(dls.train.items,groups = groups)
    group_props_V = get_group_props(dls.valid.items)
    pollock_prop_V = get_grouped_prop(dls.valid.items,groups = groups)
    d={'save_path': [save_path],'train_group_props':[group_props_T],'train_pollock_props':[pollock_prop_T],'valid_group_props':[group_props_V],'valid_pollock_props':[pollock_prop_V]}
    dist_metrics = pd.DataFrame.from_dict(data=d)
    if save_path:
        dist_metrics.to_csv(Path(save_path,write_name))
    return dist_metrics

def load_pickle(filename, folder, base_path = 'runs'):
    if not filename.endswith('.pickle'):
        filename = filename + '.pickle'
    with open(Path(base_path,folder,filename), 'rb') as f:
        pickle_file = pickle.load(f)    
    return pickle_file

def load_pickle_simple(path):
    plt = platform.system()
    if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
    if not path.endswith('.pickle'):
        path = path + '.pickle'
    with open(Path(path), 'rb') as f:
        pickle_file = pickle.load(f)    
    return pickle_file

def plot_slice_acc(painting, results,prob = True,error_bars = False,save = False,title = True,y_label = 'PMF',dpi=80):
    acc_by_group, painting_acc_at_sizes, error_group,error_painting = painting_acc_by_slice_size(results,prob = prob)
    min_size = min([int(item) for item in results.slice_size.tolist() if item.isnumeric()])
    sizes = [int(item) for item in painting_acc_at_sizes[painting].index if item.isdigit()]
    keys = [str(item) for item in range(min_size,np.max(sizes)+5,5)] + ['Max']
    data = {key:painting_acc_at_sizes[painting][key] for key in keys}
    
    names = list(data.keys())
    values = list(data.values())
    if error_bars:
        errors = {key:error_painting[painting][key] for key in keys}
        errors = list(errors.values())
    else:
        errors = list(np.zeros(len(data)))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.bar(range(len(data)), values,yerr=errors, tick_label=names)
    ax.set_xlabel('Tile Size[cm]',fontsize = 18)
    if prob:
        ylabel = y_label
    else:
        ylabel = 'accuracy'
    ax.set_ylabel(ylabel,fontsize = 18)
    if title:
        ax.set_title(painting + ' ' +ylabel, fontsize = 24)
    ax.set_ylim(0, 1)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    fig.show()
    if save:
        fig.savefig(save)
    return fig,ax


def result_paintings(results,category = ('A','B','C','D','E','F','J','G','P')):
    return list(set([item.split('_cropped')[0] for item in results.file if item.startswith(category)]))

#outdated
def run_hold_out_set(master,learn_path,
                img_path = Path('Paintings/Processed/Descreened/'),
                save_str=False, 
                Descreened = True, 
                valid_set = False,
                seed = 0,
                thread_num = False,
                computer = '',
                px_size = 256,
                grayscale = False,
                epochs = 1,
                model = resnet50,
                folders = ['10','100','105', '110', '115', '120', '125',
                         '130', '135', '140', '145', '15', '150', '155', '160',
                         '165', '170', '175', '180', '185', '190', '195', '20',
                         '200', '205', '210', '215', '220', '225', '230', '235',
                         '240', '245', '25', '250', '255', '260', '265', '270',
                         '275', '280', '285', '290', '295', '30', '300', '305',
                         '310', '315', '320', '325', '330', '335', '340', '345',
                         '35', '350', '355', '360', '40', '45', '5', '50',
                         '55', '60', '65', '70', '75', '80', '85', '90',
                         '95','Max'],
              hold_out_catalog= ['93', '165', '179', '207', '231', '801', '820', '367', '371', '380','?'],
              hold_out_files = ['A57','A66','A64','A54(R)','C20','C55','C47(R)','D19','D8','E10','F114','F61','F110','F63','F94','F87','F99','G24','G58','G94','G62','G1','G27'],
              hold_out_duplicates = ['P13(L)','P19(L)','P25(L)','P32(W)','P33(W Lowerqual)',
'P47(F Lowres)','P65(L)','P75(V)','P77(J)','P80(J)','P86(S)','P105(J)','P106(V)','P115(F Lowres)','A14(Left)'],
                remove_duplicates = True,
                notes = ''
                ):
    if valid_set:
        valid_set = valid_set
    else:
        valid_set = get_hold_out_files(master,hold_out_catalog,hold_out_files)
    # initialize learner with new dls
    dls = get_dls(master,valid_set,img_path,folders,hold_out_files=[],px_size = px_size,grayscale=grayscale)
    learn = vision_learner(dls,model,metrics=error_rate)
    #assign model run to initialized learner
    path = Path(learn_path)
    learn.path = path
    learn = learn.load('learn')
    #export valid list
    save_path = folder_management(save_str)
    valid_list = pd.DataFrame(valid_set, columns=["valid_list"])
    valid_list.to_csv(Path(save_path,'valid_list'), index=False)
    #save learner with new dls
    learn.path = save_path
    save_learner(dls,learn,save_path)
    #make and save predictions
    preds = make_predictions(learn,save_path)
    today = datetime.datetime.now()
    date_time = today.strftime("%m-%d-%Y")
    metrics = model_analysis(dls,preds,save_path = save_path,date = date_time,folders=folders,Descreened = Descreened,notes = notes,seed = seed,valid_set = valid_set,px_size = px_size,grayscale=grayscale,epochs = 1,model = model)
    return metrics

def get_learn_preds(master_or_valid_set,learn_or_path,
                    img_path = Path('Paintings/Processed/Raw/'),
                    folders = False,
                    learn_pre_path ='runs',
                    save_pre_path = 'painting_preds',
                    write_name = False,notes='',
                    overwrite=False,
                    multipath = Path('Paintings/Processed/Descreened/'),
                    item_tfms = False,
                    verbose = True):
    learn_path = 'not specified'
    plt = platform.system()
    if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
    if write_name:
        save_path = folder_management(write_name,pre_path = save_pre_path,append_date = False,overwrite=overwrite)
    if not folders:
        folders = get_folders(img_path)
    else:
        folders = folders
    if not isinstance(learn_or_path,fastai.learner.Learner):
        learn_path = learn_or_path
        learn_or_path = load_learner(Path(learn_pre_path,learn_or_path,'export.pkl'))
    if not isinstance(master_or_valid_set,list):
        master_or_valid_set = master_or_valid_set.file.tolist()
    # print_if(verbose,'img_path =',img_path)
    print_if(verbose,'folders = ',folders)
    paths = get_image_files(img_path,folders = folders)
    # print('paths',paths[0])
    # print(master_or_valid_set)
    # print([path.stem for path in paths[0:20]])
    master_or_valid_set = [item + '_' for item in master_or_valid_set]
    
    valid_set_paths = paths.filter(lambda o:o.stem.startswith(tuple(master_or_valid_set)))
    actuals = get_actuals(valid_set_paths)
    print('valid_set_paths',valid_set_paths)
    # print('actuals',actuals)
    inp,preds,dec_preds = get_new_preds(learn_or_path,valid_set_paths,item_tfms = item_tfms)
    preds_tuple = preds,actuals
    
    if write_name:
        metrics = model_analysis(valid_set_paths,preds_tuple,save_path=save_path,folders=folders,valid_set=master_or_valid_set,model=learn_path,notes=notes,computer = img_path)
        with open(Path(save_path,'preds.pickle'), 'wb') as f:
            pickle.dump(preds_tuple, f)
        if item_tfms:
            item_tfms_dict = item_tfms.to_dict()
            with open(Path(save_path,'item_tfms.json'), 'w') as f:
                json.dump(item_tfms_dict, f, indent=4)
    else:
        metrics = model_analysis(valid_set_paths,preds_tuple,save_path=False,folders=folders,valid_set=master_or_valid_set,model=learn_path,notes=notes,computer = img_path)
    # metrics = metrics.drop(['px_size', 'grayscale','hold_out_catalog','hold_out_files','hold_out_duplicates','seed','epochs','date','runtime','model','location_ran','Descreened'], axis=1)
    # metrics['base_model'] = learn_path
    # metrics['img_path'] = img_path
        
    return metrics

def apply_item_tfms(item_tfms, items): #takes in a path list of items 
    loaded_items = [cv2.cvtColor(cv2.imread(str(item)), cv2.COLOR_BGR2RGB) for item in items]
    augmented_items = [item_tfms(image = image)["image"] for image in loaded_items]
    return augmented_items

# def get_new_preds(learn, items, item_tfms=False):
#     assert len(items) > 0, 'No items exist! Check your directories!'
#     if item_tfms:
#         items = apply_item_tfms(item_tfms,items)
#     dl = learn.dls.test_dl(items, 
#                            rm_type_tfms=None, 
#                            num_workers=0)

#     dl.show_batch()  # Show the batch of items with their transformations
#     inp, preds, _, dec_preds = learn.get_preds(dl=dl, with_input=True, with_decoded=True)
#     i = getattr(learn.dls, 'n_inp', -1)
#     return inp, preds, dec_preds

def get_new_preds(learn, items, item_tfms=False):
    assert len(items) > 0, 'No items exist! Check your directories!'
    if item_tfms:
        items = apply_item_tfms(item_tfms,items)
    dl = learn.dls.test_dl(items, 
                           rm_type_tfms=None)

    # dl.show_batch()  # Show the batch of items with their transformations
    inp, preds, _, dec_preds = learn.get_preds(dl=dl, with_input=True, with_decoded=True)
    i = getattr(learn.dls, 'n_inp', -1)
    return inp, preds, dec_preds

# def get_new_preds(learn,items,item_tfms = None):
#     # print(items)
#     # item_tfms = A.Compose([
#     # # # # A.RandomCrop(width=256, height=256),
#     # A.HorizontalFlip(p=1),
#     # A.RandomBrightnessContrast(p=1),])
#     # dl = learn.dls.test_dl(items, 
#     #                        rm_type_tfms=None,
#     #                        item_tfms=item_tfms,
#     #                        num_workers=0)
#     # dl.show_batch()
#     dl = learn.dls.test_dl(items, rm_type_tfms=None, num_workers=0)
#     inp,preds,_,dec_preds = learn.get_preds(dl=dl, with_input=True, with_decoded=True)
#     i = getattr(learn.dls, 'n_inp', -1)
#     return inp,preds,dec_preds

def get_folders(path,exclude = ['Full']):
    folders = next(os.walk(path))[1]
    if isinstance(exclude,str):
        exclude = [exclude]
    folders = list(set(folders)-set(exclude))
    return folders

def get_train_test_preds(folder_str, container_folder='runs',model=resnet50):
    plt = platform.system()
    if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
    dls, learn = load_dls_learner(folder=folder_str, container_folder=container_folder,model=model)
    
    if os.path.exists(os.path.join(container_folder, folder_str, 'preds_train.pickle')):
        with open(os.path.join(container_folder, folder_str, 'preds_train.pickle'), 'rb') as f:
             preds_train = pickle.load(f) 
    else:
        preds_train = learn.get_preds(ds_idx=0)
        with open(Path(container_folder,folder_str,'preds_train.pickle'), 'wb') as f:
            pickle.dump(preds_train, f)
        
    if os.path.exists(os.path.join(container_folder, folder_str, 'preds.pickle')):
        with open(os.path.join(container_folder, folder_str, 'preds.pickle'), 'rb') as f:
             preds_valid = pickle.load(f) 
    else:
        preds_valid = learn.get_preds()
        with open(Path(container_folder,folder_str,'preds_train.pickle'), 'wb') as f:
            pickle.dump(preds_valid, f)

    train_results = get_results_df(dls,preds_train,save_path=Path(container_folder,folder_str), write_name='trainset_results.csv', TRAIN=True)
    valid_results = get_results_df(dls,preds_valid,save_path=Path(container_folder,folder_str), write_name='validset_results.csv', TRAIN=False)
    
    return train_results, valid_results

def get_train_test_ho_preds(folder_str, master,container_folder='runs',arch=resnet50):
    plt = platform.system()
    if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
    dls, learn = load_dls_learner(folder=folder_str, container_folder=container_folder,arch=arch)

    metrics = pd.read_csv(os.path.join(container_folder,folder_str,'metrics.csv'))
    ho= get_hold_out_from_metrics(master,metrics)
    folders = ['10','100','105', '110', '115', '120', '125',
        '130', '135', '140', '145', '15', '150', '155', '160',
        '165', '170', '175', '180', '185', '190', '195', '20',
        '200', '205', '210', '215', '220', '225', '230', '235',
        '240', '245', '25', '250', '255', '260', '265', '270',
        '275', '280', '285', '290', '295', '30', '300', '305',
        '310', '315', '320', '325', '330', '335', '340', '345',
        '35', '350', '355', '360', '40', '45', '50',
        '55', '60', '65', '70', '75', '80', '85', '90',
        '95','Max']
    
    if os.path.exists(os.path.join(container_folder, folder_str,'preds','HO_raw', 'results.csv')):
        print('HO results already done')
        # ho_results = pd.read_csv(os.path.join(container_folder, folder_str,'preds','HO_raw', 'results.csv'))
    else:
        get_learn_preds_wrapper(ho,learn,os.path.join(folder_str,'preds','HO_raw'),img_path = 'Paintings/Processed/Raw',overwrite=True,folders = folders,save_pre_path = container_folder,redo=True)
    
    if os.path.exists(os.path.join(container_folder, folder_str, 'preds_train.pickle')):
        with open(os.path.join(container_folder, folder_str, 'preds_train.pickle'), 'rb') as f:
             preds_train = pickle.load(f) 
    else:
        preds_train = learn.get_preds(ds_idx=0)
        with open(Path(container_folder,folder_str,'preds_train.pickle'), 'wb') as f:
            pickle.dump(preds_train, f)
        
    if os.path.exists(os.path.join(container_folder, folder_str, 'preds.pickle')):
        with open(os.path.join(container_folder, folder_str, 'preds.pickle'), 'rb') as f:
             preds_valid = pickle.load(f) 
    else:
        preds_valid = learn.get_preds()
        with open(Path(container_folder,folder_str,'preds_train.pickle'), 'wb') as f:
            pickle.dump(preds_valid, f)

    train_results = get_results_df(dls,preds_train,save_path=Path(container_folder,folder_str), write_name='trainset_results.csv', TRAIN=True)
    valid_results = get_results_df(dls,preds_valid,save_path=Path(container_folder,folder_str), write_name='validset_results.csv', TRAIN=False)
    
    return train_results, valid_results


def find_dist_of_pollocks_imitations(valid_list, master_df, TRAIN=False):
    if TRAIN:
        tmp = master_df[~master_df['file'].isin(valid_list)].copy()
    else:
        tmp = master_df[master_df['file'].isin(valid_list)].copy()
        
    tmp['Group'] = tmp['file'].str[0]
    tmp = tmp.groupby('Group').agg({'numSlices': "sum", 'file': np.count_nonzero}).reset_index()  # reset_index makes 'Group' a column rather than a row
    
    seriesI = tmp[~tmp['Group'].isin(['P', 'J'])][['numSlices', 'file']].sum()
    seriesP = tmp[tmp['Group'].isin(['P', 'J'])][['numSlices', 'file']].sum()
    
    comb_df = pd.DataFrame(index=seriesI.index, columns=['I', 'P'])
    comb_df['I'] = seriesI
    comb_df['P'] = seriesP
    
    comb_df['ratio'] = comb_df['P'] / (comb_df['P'] + comb_df['I'])
    
    return comb_df
    
    
def check_distribution_of_populations(valid_list, master_df, categories=None):
    
    if categories is None:
        categories = ['A', 'C', 'D', 'E', 'F', 'G', 'J', 'P']
        
    pop_df = pd.DataFrame(index=categories, columns=['Total_pop', 'Valid_pop'])
    
    
    for cat in categories:
        tmp = master_df[master_df['file'].str.startswith(cat)]
        
        pop_df.loc[cat, 'Total_pop'] = (tmp.shape[0])/(master_df.shape[0])
        print(f'Total Pop has {pop_df.loc[cat, "Total_pop"]:0.2f} for {cat}')
    
        cat_list = [ii for ii in valid_list if ii.startswith(cat)]
        pop_df.loc[cat, 'Valid_pop'] = len(cat_list) / len(valid_list)
        print(f'Valid Pop has {pop_df.loc[cat, "Valid_pop"]:0.2f} for {cat}')
    
    return pop_df
    
def get_actuals(paths):
    actuals = []
    for path in paths:
        actuals.append(is_pollock(path))
    integer_map = map(int, actuals)
    actuals = list(integer_map)
    return tensor(actuals)

def get_combined_preds(paths,container_folder = '',pickle = 'preds.pickle'):   
    preds_list_probs = []
    preds_list_actuals = []
    for path in paths:
        preds_path = os.path.join(container_folder,path,pickle)
        if os.path.exists(preds_path):
            preds_temp=load_pickle_simple(preds_path)
            preds_list_probs.append(preds_temp[0])
            preds_list_actuals.append(preds_temp[1])
    preds_list_probs=torch.cat(preds_list_probs)
    preds_list_actuals = torch.cat(preds_list_actuals)
    return (preds_list_probs,preds_list_actuals)

def get_combined_dfs(folder,container_folder = 'painting_preds',notes = '',img_path = 'Paintings/Processed/Descreened/'):
    folder_path = os.path.join(container_folder,folder)
    subfolders= [item for item in next(os.walk(folder_path))[1] if not item.startswith('.')]
    paths=[os.path.join(folder_path,path) for path in subfolders]
    file_list = ['preds.pickle','metrics.csv','full_paints_results.csv','results.csv']
    get_paths(folder_path,file_list)
    metrics = get_combined_df(paths,container_folder = '',csv = 'metrics.csv')
    full_paints_results = get_combined_df(paths,container_folder = '',csv = 'full_paints_results.csv')
    results = get_combined_df(paths,container_folder = '',csv = 'results.csv')
    preds = get_combined_preds(paths)
    files = results.file
    sizes = results.slice_size
    painting_paths = [Path(img_path,s,f) for s,f in zip(sizes,files)]
    full_metrics = model_analysis(painting_paths,preds,save_path=folder_path,folders=metrics.folders.iloc[0],valid_set=subfolders,model=metrics.model.iloc[0],notes=notes,computer = folder)
    return results

def get_learn_preds_wrapper(master_or_list,learner_folder,save_folder,img_path = False,overwrite=False,folders = False,save_pre_path = 'painting_preds',redo=False):
# Can actively set the img_path for specific testing: img_path = 'Paintings/Processed/Descreened/'
    if isinstance(master_or_list,pd.core.frame.DataFrame):
        file_list = master_or_list.file.to_list()
    else:
        file_list = master_or_list
    if not img_path:
        assert isinstance(master_or_list,pd.core.frame.DataFrame), 'img_path = False, dynamically setting path only works with master input'
    # print(file_list)
    for file in file_list:
        if not img_path:
            if master_or_list[master_or_list.file == file].artifacts.iloc[0] == 'True':
                img_path_set = 'Paintings/Processed/Descreened/'
            else:
                img_path_set = 'Paintings/Processed/Raw/'
        else:
            img_path_set = img_path
        path = str(Path(save_pre_path,save_folder,file))
        if redo:
            get_learn_preds([file],learner_folder,img_path = img_path_set,write_name = Path(save_folder,file),overwrite=overwrite,folders=folders,save_pre_path = save_pre_path)
        else:
            if not os.path.exists(path):
                # print('folder dont exist')
                get_learn_preds([file],learner_folder,img_path = img_path_set,write_name = Path(save_folder,file),overwrite=overwrite,folders=folders,save_pre_path = save_pre_path)
            elif len(next(os.walk(path))[2])==0:
                print(path + ' is empty. Writing to folder...')
                get_learn_preds([file],learner_folder,img_path = img_path_set,write_name = Path(save_folder,file),overwrite=overwrite,folders=folders,save_pre_path = save_pre_path)
            else:
                print(path + ' existed and had stuff in it. skipping')
    metrics = get_combined_dfs(save_folder,container_folder = save_pre_path)
    return metrics

def get_painting_confidence(painting,results,metrics,binarize=False,one_vote=False,binarize_prob=0.5, decision_prob=0.5):
    # results = pd.read_csv(Path(model_folder_path,model_folder,'results.csv'), low_memory=False)
    # metrics = pd.read_csv(Path(model_folder_path,model_folder,'metrics.csv'), low_memory=False)
    # model_confidence = 1-metrics.painting_error_rate.iloc[0]
    r = vote_system(results,one_vote=one_vote, binarize=binarize, binarize_prob=binarize_prob, decision_prob=decision_prob)
    painting_confidence = r[r.painting == painting].pollock_prob.iloc[0]
    # print(painting,painting_confidence)
    # confidence = model_confidence * painting_confidence
    return painting_confidence

def get_painting_confidence_from_folder(painting,model_folder,model_folder_path = 'runs',binarize=False,one_vote=False,binarize_prob=0.5, decision_prob=0.5):
    results = pd.read_csv(Path(model_folder_path,model_folder,'results.csv'), low_memory=False)
    metrics = pd.read_csv(Path(model_folder_path,model_folder,'metrics.csv'), low_memory=False)  
    # print(metrics.iloc[0])
    return get_painting_confidence(painting,results,metrics,one_vote=one_vote, binarize=binarize, binarize_prob=binarize_prob, decision_prob=decision_prob)

def get_painting_closest_PMF(learner_results,test_painting_result,one_vote=False, binarize=False,catagory = ('P','J')):
    r = vote_system(learner_results, one_vote=one_vote, binarize=binarize, binarize_prob=0.5, decision_prob=0.5)
    r['pp_distance'] = np.abs(r.pollock_prob-test_painting_result)
    r_catagory = r[r.painting.str.startswith(catagory)]
    return r_catagory[r_catagory.pp_distance ==min(r_catagory.pp_distance)]

def get_above_below_PMF_percentage(learner_results,test_painting_result,one_vote=False, binarize=False,catagory = ('P','J'),round_to = 2,remove_special = ['A9(right)','JPCR_01088']):
    r = vote_system(learner_results, one_vote=one_vote, binarize=binarize, binarize_prob=0.5, decision_prob=0.5)
    r = r[~r.painting.isin(remove_special)]
    r['group'] = r.painting.str[0]
    rp = r[r.group.isin(catagory)]
    above = rp[rp.pollock_prob > test_painting_result]
    below = rp[rp.pollock_prob < test_painting_result]
    percent_above = round(100*len(above)/len(rp),round_to)
    percent_below = round(100*len(below)/len(rp),round_to)
    return percent_below,percent_above

def check_train_test_no_overlap(dls):
    valid= set([item.stem.split('_c')[0].split('(')[0] for item in dls.valid.items])
    train= set([item.stem.split('_c')[0].split('(')[0] for item in dls.train.items])
    return list(valid & train)

def get_similar_min_dim_sized_painting(file,master_with_file,master_look,catagory = ('P','J')):
    M=master_with_file
    test_smallest_dim = min(M[M.file==file].height_cm.iloc[0],M[M.file==file].width_cm.iloc[0])
    M_filtered = master_look[np.logical_and(master_look.file != file,master_look.file.str.startswith(catagory))]
    mins = [min(item1,item2) for item1,item2 in zip(M_filtered.height_cm,M_filtered.width_cm)]
    min_diffs = [round(np.abs(item-test_smallest_dim),2) for item in mins]
    min_loc = min_diffs.index(min(min_diffs))
    df = M_filtered.iloc[min_loc]
    return df.file
    
def get_hold_out_from_metrics(master,metrics):
    hold_out_catalog = metrics.hold_out_catalog.apply(eval).iloc[0]
    hold_out_duplicates = metrics.hold_out_duplicates.apply(eval).iloc[0]
    hold_out_files = metrics.hold_out_files.apply(eval).iloc[0]
    hold_out_files_full = hold_out_files+hold_out_duplicates
    hold_out_files_full = get_hold_out_files(master,hold_out_catalog,hold_out_files_full)
    return hold_out_files_full

def get_max_output_size_px_based(master,file_name):
    width_cm = (master[master.file == file_name].width_px/master[master.file == file_name].px_per_cm_height).iloc[0]
    height_cm = (master[master.file == file_name].width_px/master[master.file == file_name].px_per_cm_height).iloc[0]
    return int(min(width_cm,height_cm))

def get_sorted_pollocks(stats_df,num_per_group,category = ('P','J')):
    half_num_per_group = int(num_per_group/2)
    PJ = stats_df[stats_df.painting.str.startswith(category)]
    lowest= PJ.sort_values('PMF').iloc[0:num_per_group]
    middle = PJ.sort_values('PMF').iloc[int(len(PJ)/2)-half_num_per_group:int(len(PJ)/2)+half_num_per_group]
    mean=pd.concat([PJ[PJ.PMF < PJ.PMF.mean()].sort_values('PMF').iloc[-half_num_per_group:], PJ[PJ.PMF >= PJ.PMF.mean()].sort_values('PMF').iloc[0:half_num_per_group]])
    equal_to_one = PJ[PJ.PMF == 1]
    num_equal_to_one = len(equal_to_one)
    total_num = len(PJ)
    return lowest,middle,mean,equal_to_one,num_equal_to_one,total_num

def get_inquiry(master,PJ,inquiry = ('P1(W)', 'P71(', 'P77(', 'P106(', 'P108(')):
    inquiry_titles = master[master.file.str.startswith(inquiry)].title
    inquiry_paintings = master[master.title.isin(inquiry_titles)]
    inquiry_df = PJ[PJ.painting.isin(inquiry_paintings.file)]
    return pd.merge(left = inquiry_df,right = master[['file','title']],left_on='file',right_on='file',how='left').sort_values('title')

def get_learner_results(container = 'runs',
                        learner_folder = 'gcp_classifier-3_10-Max_color_10-15-2022',
                        sets = ('train','valid','hold_out'),
                        file_strs = False #False or a list that matches sets
                        ):
    dtype_options = {'file':str,'slice_size':str,'prediction':int,'actual':int,'pollock_prob':float,'failed':bool,'painting_name':str,'group':str,'painting':str}
    results_list = []
    if file_strs:
        sets_files_dict = dict(zip(sets,file_strs))
    else:
        sets_files_dict = dict(zip(('train','valid','hold_out'),['trainset_results.csv','validset_results.csv',os.path.join('preds','HO_raw','results.csv')]))
    for painting_set in sets:
        result_temp = pd.read_csv(Path(container,learner_folder,sets_files_dict[painting_set]),dtype=dtype_options)
        result_temp['set'] = painting_set
        results_list.append(result_temp)
    # train_results = pd.read_csv(Path(container,learner_folder,'trainset_results.csv'),dtype=dtype_options)
    # train_results['set']= 'train'    
    # valid_results = pd.read_csv(Path(container,learner_folder,'validset_results.csv'),dtype=dtype_options)
    # valid_results['set'] = 'valid'
    # learner_results_ho = pd.read_csv(Path(container,learner_folder,'preds','HO_raw','results.csv'),dtype=dtype_options)
    # learner_results_ho['set']='hold_out'
    # results = pd.concat([train_results, valid_results,learner_results_ho], axis=0)

    results = pd.concat(results_list, axis=0)
    results['slice_size'] = results['slice_size'].astype(str)
    results['painting'] = results.file.str.split('_cropped').str[0].str.split('(').str[0]
    results = results[results.set.isin(sets)].reset_index()
    return results[['file','slice_size','prediction','actual','pollock_prob','failed','painting_name','group','painting','set']]

def get_item_tfms_from_dict(path, tfms_json_name = 'item_tfms.json'):
    with open(os.path.join(path,tfms_json_name), 'r') as f:
        item_tfms_dict = json.load(f)

    # Convert the dictionary back to the item_tfms object
    return A.Compose.from_dict(item_tfms_dict)

def get_albumentations_tfms(transforms_list,always_apply = True):
    if isinstance(transforms_list,str):
        transforms_list = [transforms_list]
    albumentations_tfms = []

    for transform_info in transforms_list:
        transform_components = transform_info.split(':')
        transform_name = transform_components[0]
        
        if len(transform_components) > 1:
            transform_args_str = transform_components[1]
            
            # Extract values from the string and convert to integers
            if 'limit=' in transform_args_str:
                limit_args = transform_args_str.split('=')[1]
                limit_args = tuple(map(int, limit_args.strip('()').split(',')))
                transform_args = {'limit': limit_args}
            else:
                transform_args = {}
            
            transform = getattr(A, transform_name)(**transform_args, always_apply=always_apply)
        else:
            transform = getattr(A, transform_name)(always_apply=always_apply)
            
        albumentations_tfms.append(transform)

    return A.Compose(albumentations_tfms)