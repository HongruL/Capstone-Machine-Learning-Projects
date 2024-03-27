import spacy

nlp = spacy.load('en_core_web_sm')
text = 'My system keeps crashing his crashed yesterday, ours crashes daily'
doc = nlp(text)
print([token.lemma_ for token in doc])

def lemmatize_text(text):
    doc = nlp(text)
    text = ' '.join([token.lemma_.lower() for token in doc
                     if not token.is_stop
                     and not token.is_punct
                     and not token.is_space])
    return text


if __name__ == '__main__':
    print(lemmatize_text(text))