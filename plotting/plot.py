import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='final_plots', help='Directory to save plots')
parser.add_argument('--task', type=str, default='scifact')
parser.add_argument('--dir_tables', type=str, default=['baseline_analysis'], nargs='+')
parser.add_argument('--dir_rand', default=None)

args = parser.parse_args()


def plot(df, hue = "Strategy", hue_order = None):
    sns.set(font_scale=1.25)


    for value in ['Acc', 'F1']:
        key='Instance'
        title = '{}_Few-Shot_Veracity_Classification_{}_Performance'.format(str(args.task).upper(), value)

        if len(set(df['Model'])) ==6:
            col_orders =['BERT-base', 'RoBERTa-base', 'DeBERTa-base', 'BERT-large', 'RoBERTa-large', 'DeBERTa-large']
        else:
            col_orders =None
        print(col_orders)
        #
        print(set(df['Strategy']))
        if 'Active_PETs-o' in set(df['Strategy']):
            hue_order = ['Active_PETs-o', 'Active_PETs', 'random', 'BADGE', 'ALPS', 'CAL']
        else:
            hue_order = ['Active_PETs', 'random', 'BADGE', 'ALPS', 'CAL']

        print(hue_order)

        g = sns.FacetGrid(df, hue=hue, hue_order=hue_order,
                          col='Model', col_order=col_orders,
                          margin_titles=True, legend_out=False)
        g.map(sns.lineplot, key, value)

        g.set_axis_labels('Instances', value)
        axes = g.fig.axes

        g.axes[0][0].legend(loc='upper left', fontsize=10)

        plt.savefig('{}/{}.png'.format(args.output_dir, title), dpi=800)




if __name__ == '__main__':
    model_mapping = {
               'bert-base':'BERT-base', 'roberta-base':'RoBERTa-base', 'deberta-base':'DeBERTa-base',
               'bert-large':'BERT-large', 'roberta-large':'RoBERTa-large', 'deberta-large':'DeBERTa-large'
               }
    strategy_mapping = {'alps':'ALPS',
                        'commitee_weighted_vote':'Active_PETs',
                        # 'commitee_vote': 'Ensemble_w/o_weighting',
                        'activepets': 'Active_PETs',
                        'cal': 'CAL',
                        'rand':'random',
                        'badge':'BADGE'}
    df=pd.DataFrame()
    for d in args.dir_tables:
        print(d)
        df_= pd.read_table("{}/{}.csv".format(d, args.task))
        df_['Model']=df_['Model'].map(model_mapping)
        df_ =df_[df_.Model.isin(model_mapping.values())]

        df_['Strategy']=df_['Strategy'].map(strategy_mapping)
        if 'o' in d.split('/'):
            df_['Strategy']=df_['Strategy'].map({'Active_PETs':'Active_PETs-o'})
            print(df_['Strategy'])
        print(df_)
        df= pd.concat([df, df_], ignore_index=True)
    # df.reset_index(drop=True)
    os.makedirs(args.output_dir, exist_ok=True)

    df.to_csv("{}/{}.csv".format(args.output_dir, args.task),
                                        sep='\t', encoding='utf-8', float_format='%.3f')


    print(df)

    plot(df)