model_names = ['bart', 'xlnet']
task_names = ['sst2', 'mrpc', 'qnli',]
num_seeds = 10
num_ngrams = 3
ngrams_list = list(range(1, 1+num_ngrams))
seeds_list = list(range(num_seeds))