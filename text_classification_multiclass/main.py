from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from text_preprocessing import text_normalizer as tn

# Get data
data = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers', 'footers', 'quotes'))
data_labels_map = dict(enumerate(data.target_names))

# Building the dataframe
corpus, target_labels, target_names = data.data, data.target, [data_labels_map[label] for label in data.target]
df = pd.DataFrame({'Article': corpus, 'Target Label': target_labels, 'Target Name': target_names})
df.drop(index=df[(df['Article'].str.strip() == '')].index, inplace=True)

# normalize our corpus
norm_corpus = tn.normalize_text(text_list=df['Article'])
df['Clean Article'] = norm_corpus

df.replace(r'^(\s?)+$', np.nan, regex=True, inplace=True)
df = df.dropna().reset_index(drop=True)


# train-test split
train_corpus, test_corpus, train_label_nums, test_label_nums, train_label_names, test_label_names = train_test_split(
    np.array(df['Article']),
    np.array(df['Target Label']),
    np.array(df['Target Name']),
    stratify=np.array(df['Target Label']),
    test_size=0.33, random_state=42
)

