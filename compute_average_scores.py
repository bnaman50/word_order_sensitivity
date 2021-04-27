import sys
import torch
import settings
import itertools
import statistics
import pandas as pd
import my_utils as eutils
from transformers import set_seed
from toolz.dicttoolz import merge_with
from preprocess_for_dev_s import get_arguments
from get_model_scores_per_dataset import get_model_scores

## This is for reproducibiltiy
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(0)

def get_settings_list():
    model_names = settings.model_names
    task_names = settings.task_names
    ngrams_list = settings.ngrams_list
    seeds_list = settings.seeds_list

    if sys.gettrace() is not None:
        print(f'In debugging')
        model_names = model_names[:1]
        task_names = task_names[:2]
        ngrams_list = ngrams_list[:1]
        seeds_list = seeds_list[:2]
    return model_names, task_names, ngrams_list, seeds_list

def word_order_sensitivity(accuracy):
    """
    :param accuracy: in [0, 1]
    :return:
    """
    return (1-accuracy)/0.5

def compute_dev_s_scores():
    ## TODO: Write the values to the dataframe dynamically instead of at the very end
    key_name = 'dev_s'
    model_names, task_names, ngrams_list, seeds_list = get_settings_list()
    args = get_arguments()
    args.inp_dir = eutils.mkdir_p(f'./out_dir/{key_name}')

    avg_scores = []
    for outer_idx, (task, model, ngram) in enumerate(itertools.product(task_names, model_names, ngrams_list)):
        print(f'\nKey_name: {key_name}, Model: {model}, task: {task}, ngram: {ngram}\n')
        args.task_name = task
        args.model = model
        args.ngram = ngram

        tmp_res = []
        for inner_idx, seed in enumerate(seeds_list):
            args.seed = seed  ## Set the seed
            tmp_res.append(get_model_scores(args))
        dict_of_list = merge_with(list, *tmp_res)
        dict_of_avg_val = {key: statistics.mean(val) for key, val in dict_of_list.items()}
        avg_scores.append({'model': model, 'task_name': task, 'ngram': ngram, 'word_order_sensitivity':word_order_sensitivity(dict_of_avg_val['accuracy']),
                           **dict_of_avg_val})

    df = pd.DataFrame(avg_scores)
    file_name = args.out_dir / f'{key_name}_scores.csv'
    df.to_csv(file_name, index=False)

def compute_dev_r_scores():
    key_name = 'dev_r'
    model_names, task_names, ngrams_list, seeds_list = get_settings_list()
    args = get_arguments()
    args.inp_dir = eutils.mkdir_p(f'./out_dir/{key_name}')

    avg_scores = []
    for outer_idx, (task, model) in enumerate(itertools.product(task_names, model_names)):
        print(f'\nKey_name: {key_name}, Model: {model}, task: {task}\n')
        args.task_name = task
        args.model = model

        tmp_res = get_model_scores(args)
        avg_scores.append({'model': model, 'task_name': task, **tmp_res})

    df = pd.DataFrame(avg_scores)
    file_name = args.out_dir / f'{key_name}_scores.csv'
    df.to_csv(file_name, index=False)

if __name__ == '__main__':
    compute_dev_r_scores()
    compute_dev_s_scores()
    print(f'Done')
