import tarfile, urllib, email, email.policy, re, nltk, urlextract
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from preprocessing import EmailToWordCounterTransformer, WordCounterToVectorTransformer

def fetch_spam_data():
    spam_root = "http://spamassassin.apache.org/old/publiccorpus/"
    ham_url = spam_root + "20030228_easy_ham.tar.bz2"
    spam_url = spam_root + "20030228_spam.tar.bz2"

    spam_path = Path() / "datasets" / "spam"
    spam_path.mkdir(parents=True, exist_ok=True)
    for dir_name, tar_name, url in (("easy_ham", "ham", ham_url), ("spam", "spam", spam_url)):
        if not Path(spam_path / dir_name).is_dir():
            path = (spam_path / tar_name).with_suffix(".tar.bz2")
            print("Downloading ", path)
            urllib.request.urlretrieve(url, path)
            with tarfile.open(path, "r:bz2") as tar_bz2_file:
                tar_bz2_file.extractall(path=spam_path)
    return [spam_path / dir_name for dir_name in ('easy_ham', 'spam')]

ham_dir, spam_dir = fetch_spam_data()

# Load the emails
ham_filenames = [f for f in sorted(ham_dir.iterdir()) if len(f.name) > 20]
spam_filenames = [f for f in sorted(spam_dir.iterdir()) if len(f.name) > 20]

def load_email(filepath):
    with open(filepath, 'rb') as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

ham_emails = [load_email(filepath) for filepath in ham_filenames]
spam_emails = [load_email(filepath) for filepath in spam_filenames]

def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        multipart = ", ".join([get_email_structure(sub_email) for sub_email in payload])
        return f"multipart({multipart})"
    else:
        return email.get_content_type()

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures


# Split data
X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
preprocessor = Pipeline([
    ('email_to_wordcount', EmailToWordCounterTransformer()),
    ('wordcount_to_vector', WordCounterToVectorTransformer())
])

log_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
score = cross_val_score(log_clf, X_train, y_train, cv=3)
print(score.mean())