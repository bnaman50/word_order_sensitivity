import os
import sys
import torch
import argparse
import my_utils as eutils

from collections import Counter, defaultdict

from transformers import (
    set_seed,
)
from datasets import load_dataset, concatenate_datasets

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input arguments for which dataset to select')

    ## Program Arguments
    parser.add_argument('--seed', type=int, help='Seed value. Default=0', default=0)
    parser.add_argument('--task_name', choices=['qnli', 'sst2', 'mrpc'],
                        help='Which dataset to select. Default=qnli', default='sst2')

    parser.add_argument('--model', choices=['bart', 'xlnet'],
                        help='Which model to select. Default=bart', default='xlnet')

    parser.add_argument('--out_dir',
                        help='Output dir where to save the dataset. Default=./out_dir', default='./out_dir')

    ## Parse the arguments
    args = parser.parse_args()

    args.out_dir = eutils.mkdir_p(args.out_dir)
    return  args

def step_1(dataset, args, max_allowed_ngram=3):
    """
    For tasks with either a single-sequence or a sequence-pair input, we used examples where
    the input sequence to be modified has only one sentence3 that has more than 3 tokens
    (for shuffling 3-grams to produce a sentence different from the original).
    """
    print('In Step 1')
    if args.task_name == 'sst2':
        key = 'sentence'
    elif args.task_name == 'mrpc':
        key = 'sentence1'
    elif args.task_name == 'qnli':
        key = 'question'
    else:
        print('This data is not yet implementted.')
        sys.exit(0)

    # aa = [(idx, ii) for idx, ii in enumerate(dataset[key]) if len(ii.strip().split()) < 4]

    def filter_on_len(sample, col_name, max_allowed_ngram=3):
        return True if len(sample[key].strip().split())>max_allowed_ngram else False

    return dataset.filter(filter_on_len, fn_kwargs={'col_name':key, 'max_allowed_ngram':max_allowed_ngram+1}) ##TODO: here I explicitly made it 4. Because of punctuations at end makes it 4 tokens but for dev-s, it means 3 tokens

def step_2(dataset, args):
    """
    We only select the examples that were correctly classified by the classifier.
    """
    print('In Step 2')
    eval_results, model_tuple = eutils.get_model_scores(dataset, args)

    def filter_on_eval_resuts(sample, idx, eval_results):
        return eval_results['eval_preds_equal_to_label_id'][idx]

    return dataset.filter(filter_on_eval_resuts, fn_kwargs={'eval_results':eval_results}, with_indices=True), ((eval_results,) + model_tuple)

def step_3(dataset, args):
    """
    Balance positive and negative samples by randomly removing examples from the larger-sized class
    """
    print('In Step 3')
    unq_ele_and_cts = Counter(dataset['label'])
    assert len(unq_ele_and_cts) == 2, 'There should only be two unique labels'

    key_with_max_val = max(unq_ele_and_cts, key=unq_ele_and_cts.get)
    key_with_min_val = min(unq_ele_and_cts, key=unq_ele_and_cts.get)

    extra_count = abs(unq_ele_and_cts[key_with_max_val] - unq_ele_and_cts[key_with_min_val])

    ## Divide dataset into two datasets based on their labels
    def filter_based_on_label(sample, idx, label):
        return label == sample['label']

    key_with_data = defaultdict()
    for key in unq_ele_and_cts.keys():
        key_with_data[key] = dataset.filter(filter_based_on_label, fn_kwargs={'label': key}, with_indices=True)

    ## Remove extra counts from class with more elements
    def filter_extra_samples_from_key_with_max_val(sample, idx, extra_count):
        return False if idx < extra_count else True

    key_with_data[key_with_max_val] = key_with_data[key_with_max_val].shuffle(seed=args.seed).filter(filter_extra_samples_from_key_with_max_val,
                                                                                                     fn_kwargs={'extra_count': extra_count},
                                                                                                     with_indices=True)
    ## Combine the two datasets
    return concatenate_datasets(list(key_with_data.values())).sort('idx')

def pre_process(dataset, args):
    """
    Filter the data based on 3 steps described in the papre
    """
    dataset = step_1(dataset, args)
    dataset, eval_res_tuple = step_2(dataset, args)
    return step_3(dataset, args)

def save_dataset(dataset, args):
    ## Save as CSV file (dev-r)
    out_dir = eutils.mkdir_p(args.out_dir.joinpath('dev_r'))
    file_name = os.path.join(out_dir, f'model_{args.model}_task_name_{args.task_name}.csv')
    dataset.to_csv(path_or_buf=file_name, index=False)

def make_dev_r(args):
    print(f'\nModel is {args.model} and task is {args.task_name}\n')
    val_data = load_dataset("glue", args.task_name, split='validation') ## Load orig data
    filtered_dataset = pre_process(val_data, args) ## pre-process data
    save_dataset(filtered_dataset, args) ## save data

if __name__ == '__main__':
    args = get_arguments()

    ## This is for reproducibility
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## Call the main function
    make_dev_r(args)
    print('Done')