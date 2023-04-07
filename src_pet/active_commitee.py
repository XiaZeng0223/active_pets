import json
import logging
import os

import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data import Subset, SequentialSampler, DataLoader
from src_pet.data import processors
from src_pet import setup_commitee, train, sample, cluster
from collections import Counter
logger = logging.getLogger(__name__)


def cluster_method(sampling):
    """Given the [sampling] method for active learning, return clustering function [f]
     and [condition], boolean that indicates whether sampling
    is conditioned on prior iterations"""
    if "KM" in sampling:
        f = cluster.kmeans
        condition = False
    elif "KP" in sampling:
        f = cluster.kmeans_pp
        condition = True
    elif "FF" in sampling:
        f = cluster.kcenter
        condition = True
    elif sampling == "badge":
        f = cluster.badge
        condition = False
    elif sampling == "alps":
        f = cluster.kmeans
        condition = False
    else:
        f = None
        condition = None
    return f, condition

def acquire(pool, sampled, args, models, tokenizers):
    """Sample data from unlabeled data [pool].
    The sampling method may need [args], [model], [tokenizer], or previously
    [sampled] data."""
    scores_or_vectors = sample.pool_scores_or_vectors(pool, args, model=models, tokenizer=tokenizers)
    if args.require_inverse and len(sampled)>0:
        counts = dict(Counter(pool[sampled][-1].numpy()))
        # print('counts', counts)
        #if a count is 0, we make it 0.1 to avoid zero devide issue.
        for k in [0, 1, 2]:
            if k not in counts.keys():
                counts[k]=0.1
        weights = {k: 1 / v for k, v in counts.items()}
        if os.path.isfile('{}/logits_2/eval_logits.txt'.format(args.output_dir)):
            preds = np.loadtxt('{}/logits_2/eval_logits.txt'.format(args.output_dir)).argmax(axis=1) #deberta-large is logits_2
        # print('preds', preds)
        # print('weights', weights)
        terms = torch.tensor([weights[pred] for pred in preds])
        scores_or_vectors = scores_or_vectors*terms
    clustering, condition = cluster_method(args.sampling)
    unsampled = np.delete(torch.arange(len(pool)), sampled)

    if clustering is not None:
        # cluster-based sampling method like BADGE and ALPS
        vectors = normalize(scores_or_vectors)
        centers = sampled.tolist()
        if not condition:
            # do not condition on previously chosen points
            queries_unsampled = clustering(
                vectors[unsampled], k = args.query_size
            )
            # add new samples to previously sampled list
            queries = centers + (unsampled[queries_unsampled]).tolist()
        else:
            queries = clustering(
                vectors,
                k = args.query_size,
                centers = centers
            )
        queries = torch.LongTensor(queries)
    else:
        # scoring-based methods like maximum entropy
        scores = scores_or_vectors
        _, queries_unsampled = torch.topk(scores[unsampled], args.query_size)
        queries = torch.cat((sampled, unsampled[queries_unsampled]))
    assert len(queries) == len(queries.unique()), "Duplicates found in sampling"
    assert len(queries) > 0, "Sampling method sampled no queries."
    return queries

def main():
    args = setup_commitee.get_args()
    setup_commitee.set_seed(args)
    print(args.model_name_or_path)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    # first, get already sampled points
    sampled_file = os.path.join(args.model_name_or_path[0], 'sampled.pt')
    if os.path.isfile(sampled_file):
        sampled = torch.load(sampled_file)
    else:
        sampled = torch.LongTensor([])

    # decide which model to load based on sampling method
    args.head = sample.sampling_to_head(args.sampling)
    #load both models
    model_types = args.model_type
    model_name_or_paths = args.model_name_or_path
    base_models = args.base_model

    models =[]; tokenizers =[]; datasets=[]

    for model_type, model_name_or_path, base_model in zip(model_types, model_name_or_paths, base_models):
        args.model_type = model_type
        args.model_name_or_path = model_name_or_path
        args.base_model = base_model
        model, tokenizer, _, _= setup_commitee.load_model(args)
        dataset = train.load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        logger.info(f"Already sampled {len(sampled)} examples")
        models.append(model)
        tokenizers.append(tokenizer)
        datasets.append(dataset)
    #set the args back to all involved models
    args.model_type = model_types
    args.model_name_or_path = model_name_or_paths
    args.base_model = base_models
    print(args.model_type)
    print(datasets)
    sampled = acquire(datasets[0], sampled, args, models, tokenizers)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    torch.save(sampled, os.path.join(args.output_dir, 'sampled.pt'))
    logger.info(f"Sampled {len(sampled)} examples")
    print(sampled)
    return len(sampled)

if __name__ == "__main__":
    main()
