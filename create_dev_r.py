import torch
import settings
import itertools
from transformers import set_seed
from preprocess_for_dev_r import make_dev_r, get_arguments

if __name__ == '__main__':
    model_names = settings.model_names
    task_names = settings.task_names

    args = get_arguments()

    ## This is for reproducibility
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for task, model in itertools.product(task_names, model_names):
        args.task_name = task
        args.model = model
        make_dev_r(args)

    print('Done')