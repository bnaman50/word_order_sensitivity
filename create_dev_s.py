import sys
import torch
import timeit
import settings
import itertools
from transformers import set_seed
from preprocess_for_dev_s import make_dev_s, get_arguments
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    model_names = settings.model_names
    task_names = settings.task_names
    ngrams_list = settings.ngrams_list
    seeds_list = settings.seeds_list

    if sys.gettrace() is not None:
        model_names = ['bart']
        task_names = ['sst2', 'mrpc']
        ngrams_list = [1, 2,]
        seeds_list = list(range(1))

    args = get_arguments()

    s_time = timeit.default_timer()
    for l_idx, (task, model, ngram, seed) in enumerate(itertools.product(task_names, model_names, ngrams_list, seeds_list)):
        idx_time = timeit.default_timer()
        args.task_name = task
        args.model = model
        args.ngram = ngram
        args.seed = seed

        set_seed(args.seed) ## Set the seed
        make_dev_s(args)
        print(f'\nIdx: {l_idx}, Time taken is {timeit.default_timer() - idx_time}')

    print(f'Done. Total time taken is {timeit.default_timer() - s_time}')