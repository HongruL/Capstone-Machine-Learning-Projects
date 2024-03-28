import re
from text_preprocessing.remove_tags import strip_html_tags
from text_preprocessing.expanding_contractions import expand_contractions
from text_preprocessing.lemmatization import lemmatize_text
from text_preprocessing.remove_stopwords import remove_stopwords
import unicodedata
from nltk.corpus import wordnet

def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z\s]|_|\[|\]' if remove_digits else r'[^a-zA-Z0-9\s]|_|\[|\]'
    text = re.sub(pattern, '', text)
    return text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens

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
                       "which reportedly take up the size of two tennis courts. fabrication ^^^^ establish story place iowa ^^^^^^^^ i")
    print(normalize_text([sample_text])[0])

    text = "^^^^ establish story place iowa ^^^^^^^^ i"
    print(remove_special_characters(text))