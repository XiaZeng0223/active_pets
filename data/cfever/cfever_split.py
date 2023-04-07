from sklearn.model_selection import train_test_split
import argparse
import jsonlines
import json
import pandas as pd
from collections import Counter
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='climate-fever-dataset-r1.jsonl')
parser.add_argument('--pool_size', type=str, default='small')
parser.add_argument('--output_path', type=str, default= '.')
args = parser.parse_args()

def flatten(dataset, unique=True):
    claims = []
    evidences=[]
    labels=[]
    idx=[]
    i=0
    if unique:
        for data in dataset:
            claims.append(data['claim'].strip('\"'))
            evidences.append(data['evidences'][0]['evidence'].strip('\"'))
            labels.append(data['evidences'][0]['evidence_label'])
            idx.append(i)
            i+=1
    else:
        for data in dataset:
            for evidence in data['evidences']:
                claims.append(data['claim'].strip('\"'))
                evidences.append(evidence['evidence'].strip('\"'))
                labels.append(evidence['evidence_label'])
                idx.append(i)
                i+=1
    flattened_dataset = {'claim':claims, 'evidence':evidences, 'label':labels, 'id':idx}
    print(max(([len(e.split(' ')) for e in evidences])))

    c=Counter(labels)
    min_count = min(c.values())
    print(c)
    return pd.DataFrame.from_dict(flattened_dataset), min_count

def small_pool():
    #read
    dataset = []
    with open(args.dataset, encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    # print(dataset[0], type(dataset))

    dataset, n=flatten(dataset)
    #
    # #balance
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
    #read
    dataset = []
    with open(args.dataset, encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    # print(dataset[0], type(dataset))
    dataset, n=flatten(dataset, unique=False)
    pool = pd.concat([dataset.drop(columns='id'), dev.drop(columns='id')]).drop_duplicates(keep=False).reset_index(drop=True)
    pool["id"] = pool.index

    dev =dev.drop(columns='id').reset_index(drop=True)
    dev["id"] = dev.index
    print(dev)


    print(Counter(pool.label))
    print(Counter(dev.label))

    # #write
    pool.to_json(path_or_buf="{}/big/train.jsonl".format(args.output_path), orient='records', lines=True, force_ascii=False)
    dev.to_json(path_or_buf="{}/big/dev.jsonl".format(args.output_path), orient='records', lines=True, force_ascii=False)
    pool.insert(0, 'id', pool.pop('id'))
    dev.insert(0, 'id', dev.pop('id'))

    pool.rename(columns={'id':'index'}).to_csv('big/train.tsv', sep='\t', encoding='utf-8', index=False)
    dev.rename(columns={'id':'index'}).to_csv('big/dev.tsv', sep='\t', encoding='utf-8', index=False)


def subset():
    # get a naturally imbalanced version of test set from the perfectly balanced dev set.
    # this test set is a subset of the dev set
    # they are called dev set and test set, but in practice, we don't have dev set for few-shot training. Both of them are teset sets, just different versions.
    dist={'SUPPORTS': 56, 'NOT_ENOUGH_INFO':150, 'REFUTES':20}
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


