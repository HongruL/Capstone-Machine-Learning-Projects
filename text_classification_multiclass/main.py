from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Memory
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

from text_preprocessing import text_normalizer as tn

#caching
cache_dir = 'cached_transformers'
if not Path(cache_dir).exists():
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
memory = Memory(cache_dir, verbose=0)

# Get data
data = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers', 'footers', 'quotes'))
data_labels_map = dict(enumerate(data.target_names))

# Building the dataframe
corpus, target_labels, target_names = data.data, data.target, [data_labels_map[label] for label in data.target]
df = pd.DataFrame({'Article': corpus, 'Target Label': target_labels, 'Target Name': target_names})
df.drop(index=df[(df['Article'].str.strip() == '')].index, inplace=True)
#########
df = pd.read_csv('clean_newsgroups.csv', sep=';')
df.head()
df.loc[998, 'Clean Article']
tn.normalize_text([df.loc[998, 'Clean Article']])
norm_corpus = tn.normalize_text(text_list=df['Clean Article'])
##########

# normalize our corpus
norm_corpus = tn.normalize_text(text_list=df['Article'].tolist())
df['Clean Article'] = norm_corpus

df.replace(r'^(\s?)+$', np.nan, regex=True, inplace=True)
df = df.dropna().reset_index(drop=True)
#df.head()

# train-test split
X_train, X_test, y_train, y_test = train_test_split(df['Clean Article'], df['Target Label'],
                                                    stratify=df['Target Label'], test_size=0.33, random_state=42)
#train_corpus, test_corpus, train_label_nums, test_label_nums, train_label_names, test_label_names
X_train, X_test, y_train, y_test = train_test_split(
    np.array(df['Clean Article']),
    np.array(df['Target Label']),
    stratify=np.array(df['Target Label']),
    test_size=0.33, random_state=42
)

# Feature engineering with TF-IDF model
mnb_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('mnb', MultinomialNB()),
], memory=memory)
mnb_clf.fit(train_corpus, train_label_names)
y_pred_train_mnb = mnb_clf.predict(train_corpus)

lr_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression(random_state=42))
], memory=memory)
lr_clf.fit(train_corpus, train_label_names)
y_pred_train_lr = lr_clf.predict(train_corpus)

svm_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(kernel='rbf', random_state=42))
], memory=memory)
svm_clf.fit(train_corpus, train_label_names) # slowest
y_pred_train_svm = svm_clf.predict(train_corpus)

rf_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('rf', RandomForestClassifier(random_state=42))
], memory=memory)
rf_clf.fit(train_corpus, train_label_names)
y_pred_train_rf = rf_clf.predict(train_corpus)

def get_metrics(model, y_pred):
    '''
    output a dictionary containing the model's performance metrics
    on the entire training set and of a 5-fold cross-validation
    by inputting the model and the predicted labels on the train dataset
    '''
    metrics = {
        'accuracy_train': accuracy_score(train_label_names, y_pred),
        'accuracy_validation': cross_val_score(model, train_corpus, train_label_names, scoring='accuracy', cv=3, n_jobs=-1).mean(),
        'precision_train': precision_score(train_label_names, y_pred, average='weighted'),
        'precision_validation': cross_val_score(model, train_corpus, train_label_names, cv=3, n_jobs=-1, scoring='precision_weighted').mean(),
        'recall_train': recall_score(train_label_names, y_pred, average='weighted'),
        'recall_validation': cross_val_score(model, train_corpus, train_label_names, cv=3, n_jobs=-1, scoring='recall_weighted').mean(),
        'f1_train': f1_score(train_label_names, y_pred, average='weighted'),
        'f1_validation': cross_val_score(model, train_corpus, train_label_names, cv=3, n_jobs=-1, scoring='f1_weighted').mean(),
        'auc_train': roc_auc_score(train_label_names, y_pred, average='weighted'),
        'auc_validation': cross_val_score(model, train_corpus, train_label_names, cv=3, n_jobs=-1, scoring='roc_auc_weighted').mean(),
    }
    return metrics

metrics_mnb = pd.DataFrame(pd.Series(get_metrics(mnb_clf, y_pred_train_mnb)), columns=['mnb'])
cross_val_score(mnb_clf, train_corpus, train_label_names, cv=3, n_jobs=-1, scoring='precision_weighted').mean()
roc_auc_score(train_label_names, y_pred_train_mnb)