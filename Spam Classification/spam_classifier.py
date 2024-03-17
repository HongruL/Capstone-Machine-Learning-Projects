import tarfile, urllib, email, email.policy
from pathlib import Path

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

