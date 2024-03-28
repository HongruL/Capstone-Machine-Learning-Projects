import nltk


stopwords_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text, is_lowercase=False):
    tokens = [token.strip() for token in nltk.word_tokenize(text)]
    if is_lowercase:
        filtered_tokens = [token for token in tokens if token not in stopwords_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


if __name__ == '__main__':
    sample_text = ("US unveils world's most powerful supercomputer, beats China. " 
                   "The US has unveiled the world's most powerful supercomputer called 'Summit', " 
                   "beating the previous record-holder China's Sunway TaihuLight. With a peak performance "
                   "of 200,000 trillion calculations per second, it is over twice as fast as Sunway TaihuLight, "
                   "which is capable of 93,000 trillion calculations per second. Summit has 4,608 servers, "
                   "which reportedly take up the size of two tennis courts.")
    print(remove_stopwords(sample_text, is_lowercase=True))