import pandas as pd
import matplotlib.pyplot as plt                   
import seaborn as sns
import numpy as np

#funkcja do pobierania nazw kolumn zawierających dane numeryczne
def get_numeric_columns(df)->list:
    df_numeric=df._get_numeric_data()
    numeric_columns=list(df_numeric.columns)
    print(numeric_columns)

    return numeric_columns

def make_boxplot(column_names:list,hue,data_frame):
    for i, column_name in enumerate(column_names):
        plt.figure(i)
        ax = sns.boxplot(
                          x=column_name, 
                          hue=f"{hue}",
                          data=data_frame,
                          gap=.3,
                          native_scale=True,
                          flierprops={"marker": "o"},
                          width=.5,
                          
                        
                        )
        ax.axvline(data_frame[column_name].mean(), color=".3", dashes=(2, 2))

def make_violinplot(column_names,hue:str,df):
    for i, column_name in enumerate(column_names):
        plt.figure(i)
        ax = sns.violinplot(data=df, y=column_name, hue=f"{hue}", split=True, gap=.1, inner="quart")

def make_swarmplot(column_names,hue,df):
    for i, column_name in enumerate(column_names):
        plt.figure(i)
        ax = sns.swarmplot(data=df, x=column_name, hue=f"{hue}", size=3,dodge=True,)

def make_heatmap(df):
    corr = df.corr(method='spearman')
    sns.set(style='white')
    mask = np.triu(np.ones_like(corr,dtype=bool))
    plt.figure(figsize=(20,10),dpi=150)
    sns.heatmap(corr,  cmap='magma',vmax=.3, center=0 , square=True , linewidths=.5, cbar_kws={'shrink':.8,},
            annot=True , fmt='.2f',annot_kws={'fontsize':5})

    plt.show()