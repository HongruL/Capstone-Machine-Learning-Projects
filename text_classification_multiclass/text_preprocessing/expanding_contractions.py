from contractions import CONTRACTION_MAP
import re


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile(rf"({'|'.join(contraction_mapping.keys())})", flags=re.IGNORECASE | re.DOTALL)

    def expand_match(match):
        match_ = match.group()
        first_char = match_[0]
        expanded_contraction = contraction_mapping.get(match_) \
            if match_ in contraction_mapping.keys() \
            else contraction_mapping.get(match_.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    # the function is called for each match, and the function's return value is used as the replacement string
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


if __name__ == "__main__":
    print(expand_contractions("Y'all can't expand contractions I'd think"))