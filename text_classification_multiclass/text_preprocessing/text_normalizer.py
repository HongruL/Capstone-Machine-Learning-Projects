import re
from remove_tags import strip_html_tags
from expanding_contractions import expand_contractions
from lemmatization import lemmatize_text
from remove_stopwords import remove_stopwords

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def normalize_text(text_list, html_stripping=True,
                   to_remove_digits=True, to_remove_stopwords=True,
                   to_lemmentize=True, to_expand_contractions=True,
                   lowercase=True, to_remove_special_characters=True):
    """
    normalize each document in the corpus
    """
    normalized_corpus = []
    for text in text_list:
        if html_stripping: text = strip_html_tags(text)
        if to_expand_contractions: text = expand_contractions(text)
        if lowercase: text = text.lower()
        text = re.sub(r'[\r\n]+', ' ', text)
        if to_lemmentize: text = lemmatize_text(text)
        if to_remove_special_characters:
            #special_char_pattern = re.compile(r'([{.()!-}])')
            special_char_pattern = re.compile(r'([{.(-)!}])')
            text = special_char_pattern.sub(r' \1 ', text)
            text = remove_special_characters(text, remove_digits=to_remove_digits)
        # remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        if to_remove_stopwords: text = remove_stopwords(text, is_lowercase=lowercase)

        normalized_corpus.append(text)

    return normalized_corpus

sample_text = ("US unveils world's most powerful supercomputer, beats China. " 
                   "The US has unveiled the world's most powerful supercomputer called 'Summit', " 
                   "beating the previous record-holder China's Sunway TaihuLight. With a peak performance "
                   "of 200,000 trillion calculations per second, it is over twice as fast as Sunway TaihuLight, "
                   "which is capable of 93,000 trillion calculations per second. Summit has 4,608 servers, "
                   "which reportedly take up the size of two tennis courts.")
print(normalize_text([sample_text])[0])