import spacy

nlp = spacy.load('en_core_web_sm')


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


if __name__ == '__main__':
    text = 'My system keeps crashing his crashed yesterday, ours crashes daily'
    text2 = ("US unveils world's most powerful supercomputer, beats China. " 
                       "The US has unveiled the world's most powerful supercomputer called 'Summit', " 
                       "beating the previous record-holder China's Sunway TaihuLight. With a peak performance "
                       "of 200,000 trillion calculations per second, it is over twice as fast as Sunway TaihuLight, "
                       "which is capable of 93,000 trillion calculations per second. Summit has 4,608 servers, "
                       "which reportedly take up the size of two tennis courts.")
    print(lemmatize_text(text))
    print(lemmatize_text(text2))