from sklearn.model_selection import train_test_split
import argparse
import jsonlines
import json
import pandas as pd
from collections import Counter
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dataset.jsonl')
parser.add_argument('--retrieval', type=str, default='dataset_bm25_retrieval_top3.jsonl')
parser.add_argument('--rationale', type=str, default='full')
parser.add_argument('--corpus', type=str, default='corpus.jsonl')
parser.add_argument('--pool_size', type=str, default='big')
parser.add_argument('--output_path', type=str, default= '.')
args = parser.parse_args()

def read_scifact_oracle(dataset, corpus, retrieval):
    #gold rationale setting
    label_encodings = {'SUPPORT': 'SUPPORTS', 'NOT ENOUGH INFO': 'NOT_ENOUGH_INFO', 'CONTRADICT': 'REFUTES'}
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
    dataset = [data for data in jsonlines.open(dataset)]
    retrieval = [data for data in jsonlines.open(retrieval)]

    claims = []; evidences = []; labels = []; idx=[]
    random.seed(42)
    for i, data in enumerate(dataset):
        claim = data['claim']
        doc_ids = retrieval[i]['doc_ids']
        oracle_docs = [int(idx) for idx in data['evidence'].keys()]
        for doc_id in doc_ids:
            doc = corpus[int(doc_id)]
            if doc_id in oracle_docs:
                #if the retrieved doc has oracle rationales
                indices = [s for item in data['evidence'][str(doc_id)] for s in
                           item["sentences"]]  # data['evidence'][doc_id][0]["sentences"]
                labels.append(label_encodings[data['evidence'][str(doc_id)][0]["label"]])
            else:
                # There is no gold rationales, so we randomly select some sentences.
                # 6 sentences is the maximal lenth of evidence from the other two classes, so we cap at 6 or the len of the doc.
                indices = random.sample(range(len(doc['abstract'])), k=random.randint(1, min(len(doc['abstract']), 6)))
                labels.append(label_encodings['NOT ENOUGH INFO'])    # neutral
            evidence = " ".join([corpus[int(doc_id)]['abstract'][i].replace('\n', '') for i in indices])
            evidences.append(evidence)
            claims.append(claim)
            idx.append(i)
    flattened_dataset = {'claim':claims, 'evidence':evidences, 'label':labels, 'id':idx}
    print(max(([len(e.split(' ')) for e in evidences])))
    c=Counter(labels)
    min_count = min(c.values())
    print(c)
    # print(len(labels))
    return pd.DataFrame.from_dict(flattened_dataset), min_count


def read_scifact_full(dataset, corpus, retrieval):
    #use full abstract as rationale setting
    label_encodings = {'SUPPORT': 'SUPPORTS', 'NOT ENOUGH INFO': 'NOT_ENOUGH_INFO', 'CONTRADICT': 'REFUTES'}
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
    dataset = [data for data in jsonlines.open(dataset)]
    retrieval = [data for data in jsonlines.open(retrieval)]

    claims = []; evidences = []; labels = []; idx=[]
    random.seed(42)
    for i, data in enumerate(dataset):
        claim = data['claim']
        doc_ids = retrieval[i]['doc_ids']
        oracle_docs = [int(idx) for idx in data['evidence'].keys()]
        for doc_id in doc_ids:
            if doc_id in oracle_docs:
                #if the retrieved doc has oracle rationales
                labels.append(label_encodings[data['evidence'][str(doc_id)][0]["label"]])
            else:
                # There is no gold rationales, so we randomly select some sentences.
                # 6 sentences is the maximal lenth of evidence from the other two classes, so we cap at 6 or the len of the doc.
                labels.append(label_encodings['NOT ENOUGH INFO'])    # neutral
            evidence = " ".join([sent.replace('\n', '') for sent in corpus[int(doc_id)]['abstract']])
            evidences.append(" ".join([corpus[int(doc_id)]['title'], evidence]))    #full evidence includes the title
            claims.append(claim)
            idx.append(i)
    flattened_dataset = {'claim':claims, 'evidence':evidences, 'label':labels, 'id':idx}
    print('Longest evidence sequence length:', max(([len(e.split(' ')) for e in evidences])))
    c=Counter(labels)
    min_count = min(c.values())
    print(c)
    # print(len(labels))
    return pd.DataFrame.from_dict(flattened_dataset), min_count


def small_pool():
    if args.rationale == 'oracle':
        dataset, n = read_scifact_oracle(args.dataset, args.corpus, args.retrieval)
    elif args.rationale == 'full':
        dataset, n = read_scifact_full(args.dataset, args.corpus, args.retrieval)

    # #balance
    n=150
    S = dataset[dataset.label == "SUPPORTS"].sample(n, random_state=42)
    N = dataset[dataset.label == "NOT_ENOUGH_INFO"].sample(n, random_state=42)
    C = dataset[dataset.label == "REFUTES"].sample(n, random_state=42)
    print(len(S), len(N), len(C))

    # #split
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    for set in [S, N, C]:
        train, test = train_test_split(set, test_size=0.33, random_state=42)
        val, test = train_test_split(test, test_size=0.5, random_state=42)
        df_train = pd.concat([df_train, train])
        df_val = pd.concat([df_val, val])
        df_test = pd.concat([df_test, test])
    #
    # #write
    # df_train.to_json(path_or_buf="{}/train.jsonl".format(args.output_path), orient='records', lines=True, force_ascii=False)
    # df_val.to_json(path_or_buf="{}/val.jsonl".format(args.output_path), orient='records', lines=True, force_ascii=False)
    # df_test.to_json(path_or_buf="{}/test.jsonl".format(args.output_path), orient='records', lines=True, force_ascii=False)
    # dataset.to_json(path_or_buf="{}/dataset.jsonl".format(args.output_path), orient='records', lines=True, force_ascii=False)
    return df_train, df_val, df_test

def big_pool(dev):

    if args.rationale == 'oracle':
        dataset, n = read_scifact_oracle(args.dataset, args.corpus, args.retrieval)
    elif args.rationale == 'full':
        dataset, n = read_scifact_full(args.dataset, args.corpus, args.retrieval)

    pool = pd.concat([dataset.drop(columns='id'), dev.drop(columns='id')]).drop_duplicates(keep=False).reset_index(drop=True)
    pool["id"] = pool.index

    dev =dev.drop(columns='id').reset_index(drop=True)
    dev["id"] = dev.index
    print(dev)


    print(Counter(pool.label))
    print(Counter(dev.label))

    # #write
    pool.to_json(path_or_buf="{}/train.jsonl".format(args.output_path), orient='records', lines=True, force_ascii=False)
    dev.to_json(path_or_buf="{}/dev.jsonl".format(args.output_path), orient='records', lines=True, force_ascii=False)
    pool.insert(0, 'id', pool.pop('id'))
    dev.insert(0, 'id', dev.pop('id'))

    pool.rename(columns={'id':'index'}).to_csv("{}/train.tsv".format(args.output_path), sep='\t', encoding='utf-8', index=False)
    dev.rename(columns={'id':'index'}).to_csv("{}/dev.tsv".format(args.output_path), sep='\t', encoding='utf-8', index=False)


def subset():
    # get a naturally imbalanced version of test set from the perfectly balanced dev set.
    # this test set is a subset of the dev set
    # they are called dev set and test set, but in practice, we don't have dev set for few-shot training. Both of them are teset sets, just different versions.
    dist={'SUPPORTS': 150, 'NOT_ENOUGH_INFO':124, 'REFUTES':49}
    devset = []
    with open(args.dataset, encoding='utf-8') as f:
        for line in f:
            devset.append(json.loads(line))
    devset=pd.DataFrame.from_records(devset)
    # #balance
    S = devset[devset.label == "SUPPORTS"].sample(dist['SUPPORTS'], random_state=42)
    N = devset[devset.label == "NOT_ENOUGH_INFO"].sample(dist['NOT_ENOUGH_INFO'], random_state=42)
    C = devset[devset.label == "REFUTES"].sample(dist['REFUTES'], random_state=42)
    print(len(S), len(N), len(C))
    testset = pd.concat([S, N, C])
    print(testset)
    testset.to_json(path_or_buf="{}/test.jsonl".format(args.output_path), orient='records', lines=True, force_ascii=False)
    testset.rename(columns={'id':'index'}).to_csv('test.tsv', sep='\t', encoding='utf-8', index=False)

if __name__ == '__main__':
    if args.pool_size == 'small':
        small_pool()
    elif args.pool_size == 'big':
        df_train, df_val, df_test=small_pool()
        big_pool(pd.concat([df_train, df_val, df_test]))
    elif args.pool_size =='subset':
        subset()
