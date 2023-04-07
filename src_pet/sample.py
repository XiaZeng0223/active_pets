import glob
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Subset
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, Softmax, KLDivLoss
from torch.nn.functional import one_hot
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors.dist_metrics import DistanceMetric
import pathlib
import os
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import pairwise_distances, paired_distances
from typing import Callable, Union
from collections import Counter
from scipy.stats import entropy as entropy_
from scipy.special import softmax
from src_pet.data import (
    convert_examples_to_features,
    compute_metrics,
    processors,
    output_modes
)


def sampling_to_head(sampling):
    # given [sampling] method, return head of model that is supposed to be used
    if sampling in ["alps", "bertKM", "density"]:
        head = "lm"
    else:
        head = 'sc'
    return head

def check_model_head(model, sampling):
    """Check whether [model] is correct for [sampling] method"""
    if "MaskedLM" in model.config.architectures[0]:
        model_head = "lm"
    elif "SequenceClassification" in model.config.architectures[0]:
        model_head = "sc"
    else:
        raise NotImplementedError
    sampling_head = sampling_to_head(sampling)
    return model_head == sampling_head


def load_and_embed_examples(args, model, tokenizer, evaluate=True, text = None, sub_index = None, return_plus = False, return_only_labels = False, return_logits = False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    task =args.task_name
    processor = processors[task]()

    # Load data features from cache or dataset file
    data_split = "train"
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}".format(
            data_split,
            list(filter(None, args.base_model.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            text
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        # print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s", args.data_dir)
        examples = processor.get_train_examples(args.data_dir)
        label_list = processor.get_labels()
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode='classification',
            text=text
        )
        if args.local_rank in [-1, 0]:
            print("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    #if we only need a subset of the whole dataset, e.g. obtaining the labeled set
    # print('before indexing', len(features))
    if sub_index != None:
        features=[features[index] for index in sub_index]
    # print('after indexing', len(features))
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    if return_only_labels:
        return all_labels
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    if return_plus:
        all_labeled_emb = None
        all_labeled_logits = None
        for batch in eval_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {}
            # mask_tokens() requires CPU input_ids
            if args.head == "lm":
                input_ids_cpu = batch[0].cpu().clone()
                input_ids_mask, labels = mask_tokens(input_ids_cpu, tokenizer, args)
                input_ids = input_ids_mask if args.masked else batch[0]
                input_ids = input_ids.to(args.device)
                labels = labels.to(args.device)
                inputs["input_ids"] = input_ids
                inputs["masked_lm_labels"] = labels
            elif args.head == "sc":
                inputs["input_ids"] = batch[0]
            else:
                raise NotImplementedError
            inputs["attention_mask"] = batch[1]
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            labeled_logits = model(**inputs).logits.cpu().numpy()
            if all_labeled_logits is None:
                all_labeled_logits = labeled_logits
            else:
                all_labeled_logits = np.append(all_labeled_logits, labeled_logits, axis=0)

            labeled_emb = embedding(model, inputs, args).cpu().numpy()
            if all_labeled_emb is None:
                all_labeled_emb = labeled_emb
            else:
                all_labeled_emb = np.append(all_labeled_emb, labeled_emb, axis=0)
        return torch.tensor(all_labeled_emb), torch.tensor(all_labeled_logits), all_labels
    if return_logits:
        all_labeled_logits = None
        for batch in eval_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {}
            # mask_tokens() requires CPU input_ids
            if args.head == "lm":
                input_ids_cpu = batch[0].cpu().clone()
                input_ids_mask, labels = mask_tokens(input_ids_cpu, tokenizer, args)
                input_ids = input_ids_mask if args.masked else batch[0]
                input_ids = input_ids.to(args.device)
                labels = labels.to(args.device)
                inputs["input_ids"] = input_ids
                inputs["masked_lm_labels"] = labels
            elif args.head == "sc":
                inputs["input_ids"] = batch[0]
            else:
                raise NotImplementedError
            inputs["attention_mask"] = batch[1]
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            labeled_logits = model(**inputs).logits.cpu().numpy()
            if all_labeled_logits is None:
                all_labeled_logits = labeled_logits
            else:
                all_labeled_logits = np.append(all_labeled_logits, labeled_logits, axis=0)

        return torch.tensor(all_labeled_logits), all_labels
    else:
        all_embeds = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {}
            # mask_tokens() requires CPU input_ids
            if args.head == "lm":
                input_ids_cpu = batch[0].cpu().clone()
                input_ids_mask, labels = mask_tokens(input_ids_cpu, tokenizer, args)
                input_ids = input_ids_mask if args.masked else batch[0]
                input_ids = input_ids.to(args.device)
                labels = labels.to(args.device)
                inputs["input_ids"] = input_ids
                inputs["masked_lm_labels"] = labels
            elif args.head == "sc":
                inputs["input_ids"] = batch[0]
            else:
                raise NotImplementedError

            inputs["attention_mask"] = batch[1]
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            embeds = embedding(model, inputs, args, pooling=args.pooling).cpu().numpy()
            if all_embeds is None:
                all_embeds = embeds
            else:
                all_embeds = np.append(all_embeds, embeds, axis=0)

        return all_embeds

def read_logits(args):
    '''read logits that's generated previously'''
    logits = np.loadtxt('{}/logits/eval_logits.txt'.format(args.output_dir))
    return torch.tensor(logits)

def read_multiple_logits(args):
    '''read multiple logits that are generated previously'''
    logits = []
    for i in range(len(args.model_name_or_path)):
        filename = '{}/logits_{}/eval_logits.txt'.format(args.output_dir, i)
        logits.append(torch.tensor(np.loadtxt(filename)))
    return logits

def random(inputs, args, **kwargs):
    """Random sampling by assigning uniformly random scores to all points"""
    if args.sampling_seed:
        print('random sampling with seed {}'.format(args.sampling_seed))
        torch.manual_seed(args.sampling_seed)
        scores = Uniform(0, 1).sample((inputs["input_ids"].size(0),))
        torch.manual_seed(args.seed)
    else:
        scores = Uniform(0,1).sample((inputs["input_ids"].size(0),))
    return scores

def least_conf(model, inputs, args, **kwargs):
    """Least confident sampling by assigning confident scores of label distribution for
    example when passed through [model] """

    proba = read_logits(args).softmax(dim=-1)
    scores= 1 - torch.max(proba, dim=1).values
    return scores

def margin(model, inputs, args, **kwargs):
    """
    Calculates the margin of the top-2 prediction probabilities.
    """
    proba = read_logits(args).softmax(dim=-1).cpu().numpy()
    part = np.partition(-proba, 1, axis=1)
    scores = torch.tensor(- part[:, 0] + part[:, 1])

    return scores

def entropy(model, inputs, args, **kwargs):
    """Maximum entropy sampling by assigning entropy of label distribution for
    example when passed through [model]"""
    logits = read_logits(args)
    categorical = Categorical(logits = logits)
    scores = categorical.entropy()
    return scores


def density(model, inputs, args, tokenizer, **kwargs):
    """Maximum density sampling by calculating information density for
    example when passed through [model]"""
    X = load_and_embed_examples(args, model, tokenizer, evaluate=True, text = 'both')
    similarity_mtx = 1 / (1 + pairwise_distances(X, X, metric='cosine'))
    scores = torch.tensor(similarity_mtx.mean(axis=1))
    return scores



def commitee_vote(model, inputs, args, **kwargs):
    """Commitee vote entropy. Voting sampling by calculating the vote entropy for the Committee for
    example when passed through each m in [model]"""
    # votes = []
    # for m , i in zip(model, inputs):
    #     vote = m(**i)[0].argmax(dim=1).cpu().numpy()
    #     votes.append(vote)

    votes = [logits.argmax(dim=1).numpy() for logits in read_multiple_logits(args)]
    votes = np.transpose(votes)
    p_vote = np.zeros(shape=(votes.shape[0], 3))  #3-class
    committee = args.model_name_or_path

    for vote_idx, vote in enumerate(votes):
        vote_counter = Counter(vote)
        for class_idx, class_label in enumerate([0, 1, 2]):
            p_vote[vote_idx, class_idx] = vote_counter[class_label] / len(committee)
    entr = entropy_(p_vote, axis=1)
    scores = torch.tensor(entr)
    return scores

def commitee_weighted_vote(model, inputs, args, **kwargs):
    """Commitee vote entropy. Voting sampling by calculating the vote entropy for the Committee for
    example when passed through each m in [model]"""

    votes = [logits.argmax(dim=1).numpy() for logits in read_multiple_logits(args)]
    votes = np.transpose(votes)
    p_vote = np.zeros(shape=(votes.shape[0], 3))  #3-class
    committee = args.model_name_or_path
    #introduce weighting wrt model size
    weighting = [m.config.hidden_size for m in model]
    # weighting = [m.num_parameters() for m in model]
    w_min = min(weighting)
    weighting = [w/w_min for w in weighting]


    for vote_idx, vote in enumerate(votes):
        vote_counter = Counter()
        for v, w in zip(vote, weighting):
            vote_counter.update({v: w})
        for class_idx, class_label in enumerate([0, 1, 2]):
            p_vote[vote_idx, class_idx] = vote_counter[class_label] / len(committee)
    entr = entropy_(p_vote, axis=1)
    scores = torch.tensor(entr)
    return scores

def commitee_weighted_KL(model, inputs, args, **kwargs):
    """Commitee vote entropy. Voting sampling by calculating the vote entropy for the Committee for
    example when passed through each m in [model]"""

    probas = [logits.softmax(dim=-1).numpy() for logits in read_multiple_logits(args)]

    p_vote = np.transpose(probas, axes=[1, 0, 2])
    #get consensus that's propotional with model size
    weighting = [m.config.hidden_size for m in model]
    # weighting = [m.num_parameters() for m in model]
    w_min = min(weighting)
    weighting = [w/w_min for w in weighting]

    p_consensus = np.average(p_vote, axis=1, weights=weighting)

    committee = args.model_name_or_path
    learner_KL_div = np.zeros(shape=(probas[0].shape[0], len(committee)))
    for learner_idx, _ in enumerate(committee):
        learner_KL_div[:, learner_idx] = entropy_(np.transpose(p_vote[:, learner_idx, :]), qk=np.transpose(p_consensus))

    scores = torch.tensor(np.max(learner_KL_div, axis=1))
    return scores


def commitee_KL(model, inputs, args, **kwargs):
    """Commitee vote entropy. Voting sampling by calculating the vote entropy for the Committee for
    example when passed through each m in [model]"""
    # probas = []
    # for m , i in zip(model, inputs):
    #     proba = m(**i)[0].softmax(dim=-1).cpu().numpy()
    #     probas.append(proba)

    probas = [logits.softmax(dim=-1).numpy() for logits in read_multiple_logits(args)]

    p_vote = np.transpose(probas, axes=[1, 0, 2])
    p_consensus = np.mean(p_vote, axis=1)

    committee = args.model_name_or_path
    learner_KL_div = np.zeros(shape=(probas[0].shape[0], len(committee)))
    for learner_idx, _ in enumerate(committee):
        learner_KL_div[:, learner_idx] = entropy_(np.transpose(p_vote[:, learner_idx, :]), qk=np.transpose(p_consensus))

    scores = torch.tensor(np.max(learner_KL_div, axis=1))
    return scores

def alps(model, inputs, args, **kwargs):
    """Obtain masked language modeling loss from [model] for tokens in [inputs].
    Should return batch_size X seq_length tensor.
    model is loaded as lm rather than sc for alps"""

    labels = inputs["masked_lm_labels"]
    inputs.pop("masked_lm_labels", None)
    logits = model(**inputs).logits
    batch_size, seq_length, vocab_size = logits.size()
    loss_fct = CrossEntropyLoss(reduction='none')
    loss_batched = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
    scores = loss_batched.view(batch_size, seq_length)
    return scores

def badge_gradient(model, inputs, args, **kwargs):
    """Return the loss gradient with respect to the penultimate layer for BADGE"""
    pooled_output = embedding(model, inputs, args)
    logits = model(**inputs).logits
    batch_size, num_classes = logits.size()
    softmax = Softmax(dim=1)
    probs = softmax(logits)
    preds = probs.argmax(dim=1)
    preds_oh = one_hot(preds, num_classes=num_classes)
    scales = probs - preds_oh
    grads_3d = torch.einsum('bi,bj->bij', scales, pooled_output)
    grads = grads_3d.view(batch_size, -1)
    return grads


def cal(model, inputs, args, tokenizer, **kwargs):
    """
    CAL (Contrastive Active Learning) Acquire data by choosing those with the largest KL divergence in the predictions between a candidate dpool input
     and its nearest neighbours in the training set.
    """

    # first, get already labeled points
    sampled_file = os.path.join(args.model_name_or_path, 'sampled.pt')

    if os.path.isfile(sampled_file):
        labeled_ids = torch.load(sampled_file)
    else:
        args.query_size = 100
        print('doing random sampling for initial {} samples'.format(args.query_size))
        #use random to sample the first n instances
        return random(inputs, args)
    texta_tasks = ['pubmed', 'imdb', 'sst-2', 'cola']  # 'agnews' 'wsc'
    textab_tasks = ['cfever', 'scifact', 'scifact_oracle', 'mnli', 'mnli-mm', 'sts-b', 'mrpc', 'qqp', 'qnli', 'rte', 'wnli']

    if args.task_name in texta_tasks:
        labeled_emb, labeled_logits, labeled_y = load_and_embed_examples(args=args, model=model, tokenizer=tokenizer, evaluate=True,
                                                                     text='text_a', sub_index=labeled_ids, return_plus=True)
    elif args.task_name in textab_tasks:
        labeled_emb, labeled_logits, labeled_y = load_and_embed_examples(args=args, model=model, tokenizer=tokenizer, evaluate=True,
                                                                     text='both', sub_index=labeled_ids, return_plus=True)

    neigh = KNeighborsClassifier(n_neighbors=10)  #args.num_nei=10 by default in original implementation
    neigh.fit(X=labeled_emb, y=np.array(labeled_y))
    criterion = KLDivLoss(reduction='none')
    dpool_logits = model(**inputs).logits.cpu()
    dpool_bert_emb = embedding(model, inputs, args).cpu()
    kl_scores = []
    num_adv = 0
    distances = []
    for unlab_i, candidate in enumerate(zip(dpool_bert_emb, dpool_logits)):
        # "Finding neighbours for every unlabeled data point"
        # find indices of closesest "neighbours" in train set
        distances_, neighbours = neigh.kneighbors(X=[candidate[0].numpy()], return_distance=True)
        distances.append(distances_[0])
        preds_neigh = [np.argmax(labeled_logits[n], axis=1) for n in neighbours]
        neigh_prob = F.softmax(labeled_logits[neighbours], dim=-1)
        pred_candidate = [np.argmax(candidate[1])]
        num_diff_pred = len(list(set(preds_neigh).intersection(pred_candidate)))
        if num_diff_pred > 0: num_adv += 1
        uda_softmax_temp = 1
        candidate_log_prob = F.log_softmax(candidate[1] / uda_softmax_temp, dim=-1)
        kl = np.array([torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in neigh_prob])
        kl_scores.append(kl.mean())
    kl_scores = torch.tensor(kl_scores)
    return kl_scores




def embedding(model, inputs, args, pooling='cls', **kwargs):
    """Original alps Return the pooleroutput as embedding, e.g.
     output = model.bert(**inputs)[1] for bert.
     However, it only works with bert and albert: many models don't have pooler layer, e.g. roberta, deberta.
     Here we use the [CLS] token embeddings from last_hidden_state instead:
     model.bert(**inputs)[0] returns last_hidden_state and [:, 0, :] gets the embeddings of the [CLS] token for each instance"""
    inputs.pop("masked_lm_labels", None)
    if pooling == 'cls':
        if args.model_type =='bert':
            output = model.bert(**inputs)[0][:, 0, :]
        elif args.model_type == 'roberta':
            output = model.roberta(**inputs)[0][:, 0, :]
        elif args.model_type == 'albert':
            output = model.albert(**inputs)[0][:, 0, :]
        elif args.model_type == 'deberta':
            output = model.deberta(**inputs)[0][:, 0, :]
        elif args.model_type == 'xlnet':
            output = model.transformer(**inputs)[0][:, 0, :]
        elif args.model_type == 'longformer':
            output = model.longformer(**inputs)[0][:, 0, :]
        else:
            raise NotImplementedError
    elif pooling == 'mean':
        if args.model_type =='bert':
            output = torch.mean(model.bert(**inputs)[0], 1)
        elif args.model_type == 'roberta':
            output = torch.mean(model.roberta(**inputs)[0], 1)
        elif args.model_type == 'albert':
            output = torch.mean(model.albert(**inputs)[0], 1)
        elif args.model_type == 'deberta':
            output = torch.mean(model.deberta(**inputs)[0], 1)
        elif args.model_type == 'xlnet':
            output = torch.mean(model.transformer(**inputs)[0], 1)
        elif args.model_type == 'longformer':
            output = model.longformer(**inputs)[0][:, 0, :]
        else:
            raise NotImplementedError
    elif pooling == 'max':
        if args.model_type =='bert':
            output = torch.max(model.bert(**inputs)[0], 1).values
        elif args.model_type == 'roberta':
            output = torch.max(model.roberta(**inputs)[0], 1).values
        elif args.model_type == 'albert':
            output = torch.max(model.albert(**inputs)[0], 1).values
        elif args.model_type == 'deberta':
            output = torch.max(model.deberta(**inputs)[0], 1).values
        elif args.model_type == 'xlnet':
            output = torch.max(model.transformer(**inputs)[0], 1).values
        elif args.model_type == 'longformer':
            output = model.longformer(**inputs)[0][:, 0, :]
        else:
            raise NotImplementedError
    elif pooling == 'median':
        if args.model_type =='bert':
            output = torch.median(model.bert(**inputs)[0], 1).values
        elif args.model_type == 'roberta':
            output = torch.median(model.roberta(**inputs)[0], 1).values
        elif args.model_type == 'albert':
            output = torch.median(model.albert(**inputs)[0], 1).values
        elif args.model_type == 'deberta':
            output = torch.median(model.deberta(**inputs)[0], 1).values
        elif args.model_type == 'xlnet':
            output = torch.median(model.transformer(**inputs)[0], 1).values
        elif args.model_type == 'longformer':
            output = model.longformer(**inputs)[0][:, 0, :]
        else:
            raise NotImplementedError
    return output

def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def batch_scores_or_vectors(batch, args, model, tokenizer):
    """Return scores (or vectors) for data [batch] given the active learning method"""
    if args.sampling in ['least', 'margin', 'entropy',
                         'commitee_vote', 'commitee_KL']:   #strategies that reads logits rather than generate logits
        with torch.no_grad():
            scores_or_vectors = sampling_method(args.sampling)(model=None, inputs=None, args = args)
        return scores_or_vectors
    else:
        if type(model) != list:
            model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {}
        # mask_tokens() requires CPU input_ids
        if args.head == "lm":
            input_ids_cpu = batch[0].cpu().clone()
            input_ids_mask, labels = mask_tokens(input_ids_cpu, tokenizer, args)
            input_ids = input_ids_mask if args.masked else batch[0]
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            inputs["input_ids"] = input_ids
            inputs["masked_lm_labels"] = labels
        elif args.head == "sc":
            inputs["input_ids"] = batch[0]
        else:
            raise NotImplementedError

        inputs["attention_mask"] = batch[1]
        if args.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

        with torch.no_grad():
            scores_or_vectors = sampling_method(args.sampling)(model=model, inputs=inputs, args = args, tokenizer = tokenizer)
        return scores_or_vectors

def get_scores_or_vectors(eval_dataset, args, model, tokenizer=None):
    # Returns scores or vectors needed for active learning sampling

    # assert check_model_head(model, args.sampling), "Model-sampling mismatch"
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)

    for eval_task in eval_task_names:

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        if args.sampling in ['least', 'margin', 'entropy', 'density',
                             'commitee_weighted_vote', "commitee_weighted_KL",
                             'commitee_vote', 'commitee_KL']:
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=len(eval_dataset))
        else:
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        all_scores_or_vectors = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            scores_or_vectors = batch_scores_or_vectors(batch, args, model, tokenizer)

            if all_scores_or_vectors is None:
                all_scores_or_vectors = scores_or_vectors.detach().cpu().numpy()
            else:
                all_scores_or_vectors = np.append(all_scores_or_vectors, scores_or_vectors.detach().cpu().numpy(), axis=0)

    all_scores_or_vectors = torch.tensor(all_scores_or_vectors)
    return all_scores_or_vectors

def pool_scores_or_vectors(eval_dataset, args, model, tokenizer=None):
    scores = get_scores_or_vectors(eval_dataset, args, model, tokenizer)
    return scores




def sampling_method(method):
    """Determine function [f] given name of sampling [method] for active learning"""
    SAMPLING = {
        "activepets": commitee_weighted_vote,
        "rand": random,
        "least": least_conf,
        "margin": margin,
        "entropy": entropy,
        "density": density,
        "commitee_vote": commitee_vote,
        "commitee_weighted_vote": commitee_weighted_vote,
        "commitee_KL": commitee_KL,
        "commitee_weighted_KL": commitee_weighted_KL,
        "badge": badge_gradient,
        "alps": alps,
        "cal": cal,
        "bertKM": embedding
    }
    if method in SAMPLING:
        f =  SAMPLING[method]
    else:
        raise NotImplementedError
    return f
