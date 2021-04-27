# word_order_sensitivity

## Project Motivation
Project motivation is described in [Project_Writeup.pdf](Project_Writeup.pdf). It is inspired by the [Out of Order: How important is the sequential order of words in a sentence in Natural Language Understanding tasks?](https://anhnguyen.me/project/word-order/) paper.

## Setup
I tested my code with `Python 3.8.6`.
<br \>
```
pip install -r requirements.txt
```

Please look at `settings.py` for the current tasks and models.

#### Create `dev-r` 
`CUDA_VISIBLE_DEVICES=0 python create_dev_r.py`

#### Create `dev-s` 
`CUDA_VISIBLE_DEVICES=0 python create_dev_s.py`

#### Compute word-order-sensitivity scores
`CUDA_VISIBLE_DEVICES=0 python compute_average_scores.py`


## Presentation Slides
Presentation slides can be found [here](https://docs.google.com/presentation/d/1YhvEdotTh9Qlkn1l3rl3uw2nF-ArFfM7i3IRjFGkHws/edit?usp=sharing). Key insights are described in the slides.

## TODO
1. Fine-tune `DeBerta` model on GLUE tasks.
2. Run the same experiments on SuperGlue benchmark. (Expectation: Word sensitivity should be higher since those tasks are more complex compared to Glue).
