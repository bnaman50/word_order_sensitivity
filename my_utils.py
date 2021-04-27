import pathlib
import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
from datasets import load_metric
from scipy.special import softmax

_Model_Names = [
    'textattack/facebook-bart-large-SST-2',
    'textattack/facebook-bart-large-MRPC',
    'textattack/facebook-bart-large-QNLI',
    'textattack/xlnet-base-cased-SST-2',
    'textattack/xlnet-base-cased-MRPC',
    'textattack/xlnet-base-cased-QNLI',
]

task_to_keys = {
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "sst2": ("sentence", None),
}

coloumn_for_ngrams = {
    "mrpc": "sentence1",
    "qnli": "question",
    "sst2": "sentence",
}

def mkdir_p(inp_path):
    inp_path = pathlib.Path(inp_path).resolve()
    inp_path.mkdir(parents=True, exist_ok=True)
    return inp_path

def load_model(args, num_labels):
    ## get exact model_name
    if args.task_name == 'sst2':
        tk_name = 'sst-2'
    else:
        tk_name = args.task_name

    model_name = [ii for ii in _Model_Names if all(x in ii.lower() for x in [args.model, tk_name])]
    assert len(model_name) == 1, 'Something is wrong with model loading'
    model_name = model_name[0]

    ## Config, tokenizer, model
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    return model_name, config, model, tokenizer

def get_model_scores(dataset, args):
    # label_list = dataset.features["label"].names
    num_labels = 2 #len(label_list)
    model_name, config, model, tokenizer = load_model(args, num_labels=num_labels)
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    padding = False  # I should do this
    model_max_length = min(512, tokenizer.model_max_length)

    def preprocess_sample(examples):
        # Tokenize the texts
        inp_args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*inp_args, padding=padding, truncation=True, max_length=model_max_length, )
        return result

    dataset = dataset.map(preprocess_sample, batched=True)
    metric = load_metric("glue", args.task_name)

    def compute_metrics(p):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        probs = np.amax(softmax(logits, axis=1), axis=1)
        pred_labels = np.argmax(logits, axis=1)
        result = metric.compute(predictions=pred_labels, references=p.label_ids)  ## accuracy
        result['preds_equal_to_label_id'] = (pred_labels == p.label_ids).tolist()
        # result['avg_confidence_score_for_correct_preds'] = np.mean(scores[result['preds_equal_to_label_id']])
        result['avg_confidence_score'] = np.mean(probs)
        return result

    training_args = TrainingArguments(
        output_dir='./tmp_dir',
        per_device_eval_batch_size=64,  # batch size for evaluation
        # do_train=False, #default is false
        # do_eval=False, #it is false since default evaluations strategey is false
        logging_strategy='no',
        save_strategy='no',
        eval_accumulation_steps=1,
        skip_memory_metrics=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    eval_results = {}
    tasks = [args.task_name]
    eval_datasets = [dataset]

    for eval_dataset, task in zip(eval_datasets, tasks):
        eval_result = trainer.evaluate(eval_dataset=eval_dataset,)
        eval_results.update(eval_result)

    return eval_results, (model, tokenizer, model_name, config, num_labels) ## TODO: return as a dict
