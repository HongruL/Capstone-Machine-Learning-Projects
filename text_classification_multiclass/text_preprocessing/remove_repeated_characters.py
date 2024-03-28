import re
from nltk.corpus import wordnet


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(token) for token in tokens]
    return correct_tokens

if __name__ == '__main__':
    tokens = ['hhappyy', 'unhapppy', 'finnalllyy', 'unhapppy']
    print(remove_repeated_characters(tokens))
