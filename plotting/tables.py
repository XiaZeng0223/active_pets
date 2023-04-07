import numpy as np
from numpy import dot, mean, absolute
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import glob, os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='baseline_pets')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--output_dir', type=str, default='baseline_analysis')   #ensemble_analysis
parser.add_argument('--task', type=str, default='cfever')
parser.add_argument('--ensemble', action='store_true')


args = parser.parse_args()

def get():
    model=[]
    instance = []
    strategy = []
    acc = []
    f1=[]
    for filename in glob.glob('{}/{}/{}/*/*/results.json'.format(args.results_dir, args.seed, args.task)):
        print(filename)
        instance.append(int(filename.split('/')[-2].split('_')[-1]))
        strategy.append(filename.split('/')[-2].split('_')[0])
        model.append(filename.split('/')[-3])

        with open(filename, 'r') as f:  # open in readonly mode
            text = f.read()
        acc.append(json.loads(text)['test_set_after_training']['acc'])
        f1.append(json.loads(text)['test_set_after_training']['f1-macro'])
    df = pd.DataFrame(
        {'Model':model,
         'Strategy':strategy,
         'Instance':instance,
         'Acc':acc,
         'F1':f1,
         }).sort_values('Instance')
    df =df.reset_index().drop(columns=['index'])
    print(df)
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv("{}/{}.csv".format(args.output_dir, args.task),
                                        sep='\t', encoding='utf-8', float_format='%.3f')

def get_ensemble():
    model_mapping = {'model_3':'bert-base', 'model_4':'roberta-base', 'model_5':'deberta-base',
                     'model_0':'bert-large', 'model_1': 'roberta-large', 'model_2':'deberta-large'}
    model=[]
    instance = []
    strategy = []
    acc = []
    f1=[]
    for filename in glob.glob('{}/{}/{}/*/model_*/results.json'.format(args.results_dir, args.seed, args.task)):
        print(filename)
        instance.append(int(filename.split('/')[-3].split('_')[-1]))
        strategy.append("_".join(filename.split('/')[-3].split('_')[:-1]))
        model.append(model_mapping[filename.split('/')[-2]])

        with open(filename, 'r') as f:  # open in readonly mode
            text = f.read()
        acc.append(json.loads(text)['test_set_after_training']['acc'])
        f1.append(json.loads(text)['test_set_after_training']['f1-macro'])
    df = pd.DataFrame(
        {'Model':model,
         'Strategy':strategy,
         'Instance':instance,
         'Acc':acc,
         'F1':f1,
         }).sort_values('Instance')
    df =df.reset_index().drop(columns=['index'])
    print(df)
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv("{}/{}.csv".format(args.output_dir, args.task),
                                        sep='\t', encoding='utf-8', float_format='%.3f')

if __name__ == '__main__':
    if args.ensemble:
        get_ensemble()
    else:
        get()