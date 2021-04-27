import os
import sys
import glob
import math
import string
import random
import argparse
import itertools
import numpy as np
import my_utils as eutils
from datasets import load_dataset


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input arguments for which dataset to select')

    ## Program Arguments
    parser.add_argument('--seed', type=int, help='Seed value. Default=0', default=0)
    parser.add_argument('--ngram', type=int, help='n-gram used for randomization. Default=1',
                        default=1, choices=[1, 2, 3])
    parser.add_argument('--task_name', choices=['qnli', 'sst2', 'mrpc'],
                        help='Which dataset to select. Default=qnli', default='sst2')
    parser.add_argument('--model', choices=['bart', 'xlnet'],
                        help='Which model to select. Default=bart', default='xlnet')
    parser.add_argument('--inp_dir',
                        help='Inp dir where dev-r exists. Default=./out_dir/dev_r', default='./out_dir/dev_r')
    parser.add_argument('--out_dir',
                        help='Output dir where to save the dataset. Default=./out_dir', default='./out_dir')
    ## Parse the arguments
    args = parser.parse_args()
    args.inp_dir = eutils.mkdir_p(args.inp_dir)
    args.out_dir = eutils.mkdir_p(args.out_dir)
    return args

def _get_shuffled_version(chunks):
    if len(chunks) <= 3:
        tot_permutations = math.factorial(len(chunks))
        all_perm_gen = itertools.permutations(chunks)
        random_perm_idx = np.random.randint(low=1, high=tot_permutations) # low (inclusive) to high (exclusive)
        return next(itertools.islice(all_perm_gen, random_perm_idx, None))
    else:
        sfle = random.sample(chunks, len(chunks))
        if sfle == chunks:
            sfle = _get_shuffled_version(chunks)
        return sfle

def get_chunks(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())

def _randomize_based_on_ngram(sample, idx, col_name, ngram):
    orig = sample[col_name]
    inp_str = orig.strip()

    last_char = ''
    if inp_str[-1] in string.punctuation:
        last_char = inp_str[-1]
        inp_str = inp_str[:-1].strip()
        # if inp_str[-1] in string.punctuation:
        #     ## TODO: multiple punctuations at the end. Need to think how to handle it. VERY TRICKY
        #     aa = 1

    tokens = inp_str.strip().split()
    chunks = list(get_chunks(tokens, ngram))
    shuffle_version = _get_shuffled_version(chunks)
    flat_list = list(sum(shuffle_version, ()))
    sample[col_name] = ' '.join(flat_list) + last_char
    return sample

def randomize_ngrams(dataset, args):
    if sys.gettrace() is not None:
        orig_dataset = dataset

    col_name = eutils.coloumn_for_ngrams[args.task_name]
    dataset = dataset.map(_randomize_based_on_ngram, with_indices=True,
                          fn_kwargs={'col_name': col_name,
                                     'ngram': args.ngram})
    return dataset

def load_dataset_from_csv(args):
    ## Load the dev-r data
    datafiles_list = [file for file in glob.glob(os.path.join(args.inp_dir, f"*.csv"))]
    file_name = [ii for ii in datafiles_list if all(x in ii.lower() for x in [args.model, args.task_name])]
    assert len(file_name) == 1, 'Input File does not exist'
    dataset = load_dataset('csv', data_files=file_name[0])['train']
    return dataset

def save_dataset(dataset, args):
    ## Save as CSV file (dev-s)
    out_dir = eutils.mkdir_p(args.out_dir.joinpath(f'dev_s/seed_{args.seed}/ngram_{args.ngram}/'))
    file_name = os.path.join(out_dir, f'model_{args.model}_task_name_{args.task_name}_seed_{args.seed}_ngram_{args.ngram}.csv')
    dataset.to_csv(path_or_buf=file_name, index=False)

def make_dev_s(args):
    print(f'\nModel: {args.model}, task_name: {args.task_name}, '
          f'ngram: {args.ngram}, seed: {args.seed}\n')
    dataset = load_dataset_from_csv(args) ## Load the dev-r data
    dataset = randomize_ngrams(dataset, args) ## Randomize the dataset based on ngram
    save_dataset(dataset, args) ## save the dataset

if __name__ == '__main__':
    args = get_arguments()

    # if sys.gettrace() is not None:
    #     args.ngram = 3

    make_dev_s(args)
    print('Done')