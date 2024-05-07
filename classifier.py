import pandas as pd
from sklearn.linear_model import LogisticRegression
import fastai.vision.all as fvisall
import os 

import utils_fastai
import util


def logistic_classifier(folder_str, container_folder='runs',model=fvisall.resnet34):
    
    if os.path.exists(os.path.join(container_folder, folder_str, 'trainset_results.csv')):
        train = pd.read_csv(os.path.join(container_folder, folder_str, 'trainset_results.csv'))
        test = pd.read_csv(os.path.join(container_folder, folder_str, 'validset_results.csv'))
    else:
        train, test = utils_fastai.get_train_test_preds(folder_str=folder_str, container_folder='runs',model=model)
    
    # by slice sizes
    feature_train_by_size = util.make_feature_samples_by_slice_size(train, painting_col='painting_name', group_col='slice_size', prob_col='pollock_prob')
    category_train = util.make_category_for_each_sample(train, feature_train_by_size)
    
    
    feature_test_by_size = util.make_feature_samples_by_slice_size(test, painting_col='painting_name', group_col='slice_size', prob_col='pollock_prob')
    category_test = util.make_category_for_each_sample(test, feature_test_by_size, sample_col='painting_name')
    
    
    if len(feature_train_by_size.columns) != len(feature_test_by_size):
        for col in feature_train_by_size.columns:
            if col not in feature_test_by_size.columns:
                feature_test_by_size[col] = 0 

    clf = LogisticRegression(random_state=0).fit(feature_train_by_size, category_train)
    preds_by_sizes = clf.predict(feature_test_by_size)
    
    comp_df = pd.DataFrame(columns=['actual', 'predicted', 'failed'])
    comp_df['actual'] = category_test
    comp_df['predicted'] = preds_by_sizes
    comp_df['failed'] = comp_df['actual'] != comp_df['predicted']
    return comp_df