import sys
import copy
from preprocess_for_dev_s import get_arguments, load_dataset_from_csv
import my_utils as eutils

def load_dataset_from_csv_with_key(args, data_key):
    if data_key == 'dev_r':
        return load_dataset_from_csv(args)
    elif data_key == 'dev_s':
        args_copy = copy.deepcopy(args)
        args_copy.inp_dir = args_copy.inp_dir / f'seed_{args.seed}' / f'ngram_{args.ngram}'
        return load_dataset_from_csv(args_copy)
    else:
        print(f'Not yet implemented')
        sys.exit(0)

def get_model_scores(args):
    dataset = load_dataset_from_csv_with_key(args, args.inp_dir.stem)
    eval_results, model_tuple = eutils.get_model_scores(dataset, args)
    return {'accuracy': eval_results['eval_accuracy'],
            'avg_confidence_score': eval_results['eval_avg_confidence_score']}

if __name__ == '__main__':
    args = get_arguments()

    if sys.gettrace() is not None:
        args.inp_dir = eutils.mkdir_p('./out_dir/dev_s')
        args.out_dir = args.inp_dir

    aa = get_model_scores(args)
    print(f'Done')