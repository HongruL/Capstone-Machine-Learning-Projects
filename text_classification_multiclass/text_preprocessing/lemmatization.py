import spacy

nlp = spacy.load('en_core_web_sm')
text = 'My system keeps crashing his crashed yesterday, ours crashes daily'


def lemmatize_text(text):
    doc = nlp(text)
    text = ' '.join([token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc])
    return text


if __name__ == '__main__':
    print(lemmatize_text(text))