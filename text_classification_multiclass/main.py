from sklearn.datasets import fetch_20newsgroups
import numpy as np
from scipy.stats import loguniform
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

# Hyperparameter tuning with Randomized Search on MultinomialNB and Logistic regression model
# Specify the parameter distributions
mnb_param_dist = {
    'mnb__alpha': loguniform(1e-5, 1e0)
}
random_search_mnb = RandomizedSearchCV(estimator=mnb_clf, param_distributions=mnb_param_dist,
                                       n_iter=50, n_jobs=-1, random_state=42, cv=3)
random_search_mnb.fit(X_train, y_train)
tuned_mnb = random_search_mnb.best_estimator_
y_pred_train_tuned_mnb = tuned_mnb.predict(X_train)
y_score_tuned_mnb = cross_val_predict(tuned_mnb, X_train, y_train, cv=3, n_jobs=-1, method='predict_proba')

lr_param_dist = {
    'lr__C': loguniform(1e-4, 1e2)
}
random_search_lr = RandomizedSearchCV(estimator=lr_clf, param_distributions=lr_param_dist,
                                      n_iter=100, n_jobs=-1, random_state=42, cv=3)
random_search_lr.fit(X_train, y_train)
tuned_lr = random_search_lr.best_estimator_
y_pred_train_tuned_lr = tuned_lr.predict(X_train)
y_score_tuned_lr = cross_val_predict(tuned_lr, X_train, y_train, cv=3, n_jobs=-1, method='predict_proba')

# Evaluate tuned models
metrics_df.append(pd.Series(get_metrics(tuned_mnb, y_pred_train_tuned_mnb, y_score_tuned_mnb), name='tuned mnb').to_frame())
metrics_df.append(pd.Series(get_metrics(tuned_lr, y_pred_train_tuned_lr, y_score_tuned_lr), name='tuned lr').to_frame())

metrics_compare = pd.concat(metrics_df, axis=1)
print(metrics_compare)

# Bagging
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone
voting_clf = VotingClassifier([
    ('tuned_mnb', clone(tuned_mnb)),
    ('tuned_lr', clone(tuned_lr))
], voting='soft')
voting_clf.fit(X_train, y_train)
y_pred_train_voting_clf = voting_clf.predict(X_train)
y_score_voting_clf = cross_val_predict(voting_clf, X_train, y_train, cv=3, n_jobs=-1, method='predict_proba')
metrics_df.append(pd.Series(get_metrics(voting_clf, y_pred_train_voting_clf, y_score_voting_clf), name='voting clf').to_frame())

metrics_compare = pd.concat(metrics_df, axis=1)
print(metrics_compare)

# Test
y_pred_test_mnb = tuned_mnb.predict(X_test)
y_pred_test_lr = tuned_lr.predict(X_test)
y_pred_test_voting_clf = voting_clf.predict(X_test)

test_results = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}
for clf in (tuned_mnb, tuned_lr, voting_clf):
    test_results['accuracy'].append(accuracy_score(y_test, clf.predict(X_test)))
    test_results['precision'].append(precision_score(y_test, clf.predict(X_test), average='weighted'))
    test_results['recall'].append(recall_score(y_test, clf.predict(X_test), average='weighted'))
    test_results['f1'].append(f1_score(y_test, clf.predict(X_test), average='weighted'))

test_results_df = pd.DataFrame(test_results.values(), index=test_results.keys(), columns=['tuned mnb', 'tuned lr', 'voting clf'])
print(test_results_df)
