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
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")

from text_preprocessing import text_normalizer as tn

# caching
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
norm_corpus = tn.normalize_text(text_list=df['Clean Article'].tolist())
##########

# normalize our corpus
norm_corpus = tn.normalize_text(text_list=df['Article'].tolist())
df['Clean Article'] = norm_corpus

df.replace(r'^(\s?)+$', np.nan, regex=True, inplace=True)
df = df.dropna().reset_index(drop=True)
# df.head()

# train-test split
X_train, X_test, y_train, y_test = train_test_split(df['Clean Article'], df['Target Label'],
                                                    stratify=df['Target Label'], test_size=0.33, random_state=42)

# Feature engineering with TF-IDF model
mnb_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('mnb', MultinomialNB()),
], memory=memory)
mnb_clf.fit(X_train, y_train)
y_pred_train_mnb = mnb_clf.predict(X_train)
y_score_mnb = cross_val_predict(mnb_clf, X_train, y_train, cv=3, n_jobs=-1, method='predict_proba')

lr_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression(random_state=42))
], memory=memory)
lr_clf.fit(X_train, y_train)
y_pred_train_lr = lr_clf.predict(X_train)
y_score_lr = cross_val_predict(lr_clf, X_train, y_train, cv=3, n_jobs=-1, method='predict_proba')

svm_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(kernel='rbf', random_state=42, probability=True))
], memory=memory)
svm_clf.fit(X_train, y_train)  # slowest
y_pred_train_svm = svm_clf.predict(X_train)
y_score_svm = cross_val_predict(svm_clf, X_train, y_train, cv=3, n_jobs=-1, method='predict_proba')

rf_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('rf', RandomForestClassifier(random_state=42))
], memory=memory)
rf_clf.fit(X_train, y_train)
y_pred_train_rf = rf_clf.predict(X_train)
y_score_rf = cross_val_predict(rf_clf, X_train, y_train, cv=3, n_jobs=-1, method='predict_proba')


def get_metrics(model, y_pred, y_score):
    '''
    output a dictionary containing the model's performance metrics
    on the entire training set and of a 5-fold cross-validation
    by inputting the model and the predicted labels on the train dataset
    '''
    metrics = {
        'accuracy_train': accuracy_score(y_train, y_pred),
        'accuracy_validation': cross_val_score(model, X_train, y_train, scoring='accuracy', cv=3, n_jobs=-1).mean(),
        'precision_train': precision_score(y_train, y_pred, average='weighted'),
        'precision_validation': cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1,
                                                scoring='precision_weighted').mean(),
        'recall_train': recall_score(y_train, y_pred, average='weighted'),
        'recall_validation': cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1,
                                             scoring='recall_weighted').mean(),
        'f1_train': f1_score(y_train, y_pred, average='weighted'),
        'f1_validation': cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1, scoring='f1_weighted').mean(),
        'roc_auc_score': roc_auc_score(y_train, y_score, average='weighted', multi_class='ovr'),
    }
    return metrics


metrics_df = []
for clf, y_pred_train, y_score in [(mnb_clf, y_pred_train_mnb, y_score_mnb),
                                   (lr_clf, y_pred_train_lr, y_score_lr),
                                   (svm_clf, y_pred_train_svm, y_score_svm),
                                   (rf_clf, y_pred_train_rf, y_score_rf)]:
    col_name = list(clf.named_steps.keys())[1]
    metrics_df.append(pd.Series(get_metrics(clf, y_pred_train, y_score), name=col_name).to_frame())

metrics_compare = pd.concat(metrics_df, axis=1)
print(metrics_compare)

# try_lr_clf = lr_clf = Pipeline([
#     ('tfidf', TfidfVectorizer()),
#     ('svd', TruncatedSVD(n_components=100, random_state=42)),
#     ('lr', LogisticRegression(random_state=42))
# ])
# try_lr_clf.fit(X_train, y_train)
# try_y_pred_train_lr = try_lr_clf.predict(X_train)
# try_y_score_lr = cross_val_predict(try_lr_clf, X_train, y_train, cv=3, n_jobs=-1, method='predict_proba')
# print(pd.Series(get_metrics(try_lr_clf, try_y_pred_train_lr, try_y_score_lr)))

# try_lr_clf = lr_clf = Pipeline([
#     ('tfidf', TfidfVectorizer()),
#     ('feature_selection', SelectFromModel(RandomForestClassifier(random_state=42))),
#     ('lr', LogisticRegression(random_state=42))
# ])
# try_lr_clf.fit(X_train, y_train)
# try_y_pred_train_lr = try_lr_clf.predict(X_train)
# try_y_score_lr = cross_val_predict(try_lr_clf, X_train, y_train, cv=3, n_jobs=-1, method='predict_proba')
# print(pd.Series(get_metrics(try_lr_clf, try_y_pred_train_lr, try_y_score_lr)))

# Randomized Search
