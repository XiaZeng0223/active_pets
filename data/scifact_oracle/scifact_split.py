from sklearn.model_selection import train_test_split
import argparse
import jsonlines
import json
import pandas as pd
from collections import Counter
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dataset.jsonl')
parser.add_argument('--corpus', type=str, default='corpus.jsonl')
parser.add_argument('--pool_size', type=str, default='big')
parser.add_argument('--output_path', type=str, default= '.')
args = parser.parse_args()

def read_scifact(dataset, corpus, unique=False):
    #gold abstract setting
    label_encodings = {'SUPPORT': 'SUPPORTS', 'NOT ENOUGH INFO': 'NOT_ENOUGH_INFO', 'CONTRADICT': 'REFUTES'}
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
    dataset = jsonlines.open(dataset)
    claims = []; evidences = []; labels = []; idx=[]
    i=0
    random.seed(42)
    for data in dataset:
        claim = data['claim']
        if data['evidence']=={}:
            for doc_id in data['cited_doc_ids']:
                doc = corpus[int(doc_id)]
                # There is no gold rationales, so we randomly select some sentences.
                # 6 sentences is the maximal lenth of evidence from the other two classes, so we cap at 6 or the len of the doc.
                indices = random.sample(range(len(doc['abstract'])), k=random.randint(1, min(len(doc['abstract']), 6)))
                # print(indices)
                evidence = " ".join([corpus[int(doc_id)]['abstract'][i].replace('\n', '') for i in indices])
                evidences.append(evidence)
                labels.append(label_encodings['NOT ENOUGH INFO'])    # neutral
                claims.append(claim)
                idx.append(i)
        else:
            if unique:
                #if we only use each claim once, use the first abstract
                doc_ids = list(data['evidence'].keys())[:1]
            else:
                doc_ids =data['evidence'].keys()
            for doc_id in doc_ids:
                indices = [s for item in data['evidence'][doc_id] for s in item["sentences"]] # data['evidence'][doc_id][0]["sentences"]
                # print(indices)
                evidence = ' '.join([corpus[int(doc_id)]['abstract'][i].replace('\n', '') for i in indices])
                evidences.append(evidence)
                claims.append(claim)
                labels.append(label_encodings[data['evidence'][doc_id][0]["label"]])
                idx.append(i)
    flattened_dataset = {'claim':claims, 'evidence':evidences, 'label':labels, 'id':idx}
    print(max(([len(e.split(' ')) for e in evidences])))
    c=Counter(labels)
    min_count = min(c.values())
    print(c)
    # print(len(labels))
    return pd.DataFrame.from_dict(flattened_dataset), min_count


def small_pool():
    dataset, n = read_scifact(args.dataset, args.corpus, unique=False)

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

    dataset, n = read_scifact(args.dataset, args.corpus, unique=False)

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
