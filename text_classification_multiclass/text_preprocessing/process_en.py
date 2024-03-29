import nltk
import spacy
import unicodedata
from text_preprocessing.contractions import CONTRACTION_MAP
import re
from nltk.corpus import wordnet
from bs4 import BeautifulSoup

stopwords_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=' ')
    stripped_text = re.sub(r'[\r\n]+', ' ', stripped_text)
    return stripped_text


def lemmatize_text(text):
    doc = nlp(text)
    processed_tokens = []
    for token in doc:
        # Check if token is an entity and should be preserved
        if token.ent_type_:
            processed_tokens.append(token.text.lower())
        elif not token.is_stop and not token.is_punct and not token.is_space:
            processed_tokens.append(token.lemma_.lower())

    return ' '.join(processed_tokens)


# def remove_repeated_characters(tokens):
#     repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
#     match_substitution = r'\1\2\3'
#
#     def replace(old_word):
#         if wordnet.synsets(old_word):
#             return old_word
#         new_word = repeat_pattern.sub(match_substitution, old_word)
#         return replace(new_word) if new_word != old_word else new_word
#
#     correct_tokens = [replace(token) for token in tokens]
#     return correct_tokens

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile(r'({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(match):
        match_ = match.group()
        first_char = match_[0]
        expanded_contraction = contraction_mapping.get(match_, contraction_mapping.get(match_.lower()))
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z\s]|_|\[|\]' if remove_digits else r'[^a-zA-Z0-9\s]|_|\[|\]'
    text = re.sub(pattern, '', text)
    return text


# def remove_stopwords(text, is_lowercase=False):
#     tokens = [token.strip() for token in nltk.word_tokenize(text)]
#     if is_lowercase:
#         filtered_tokens = [token for token in tokens if token not in stopwords_list]
#     else:
#         filtered_tokens = [token for token in tokens if token.lower() not in stopwords_list]
#     filtered_text = ' '.join(filtered_tokens)
#     return filtered_text

def normalize_text(text_list, html_stripping=True, accented_char_removal=True,
                   to_remove_digits=True, to_lemmentize=True,
                   to_expand_contractions=True, to_remove_special_characters=True):
    """
    normalize each document in the corpus
    """
    normalized_corpus = []
    for text in text_list:
        # Strip HTML
        if html_stripping: text = strip_html_tags(text)
        # Remove accented characters
        if accented_char_removal: text = remove_accented_chars(text)
        # Expand contractions
        if to_expand_contractions: text = expand_contractions(text)
        # Remove special characters and\or digits
        if to_remove_special_characters:
            # special_char_pattern = re.compile(r'([{.(-)!}])')
            # # insert spaces between special characters to isolate them
            # text = special_char_pattern.sub(r' \1 ', text)
            text = remove_special_characters(text, remove_digits=to_remove_digits)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Lemmatize text
        if to_lemmentize: text = lemmatize_text(text)

        normalized_corpus.append(text)

    return normalized_corpus


if __name__ == '__main__':
    sample_text = ("US unveils world's most powerful supercomputer, beats China. "
                   "The US has unveiled the world's most powerful supercomputer called 'Summit', "
                   "beating the previous record-holder China's Sunway TaihuLight. With a peak performance "
                   "of 200,000 trillion calculations per second, it is over twice as fast as Sunway TaihuLight, "
                   "which is capable of 93,000 trillion calculations per second. Summit has 4,608 servers, "
                   "which reportedly take up the size of two tennis courts.")
    print(normalize_text([sample_text])[0])
