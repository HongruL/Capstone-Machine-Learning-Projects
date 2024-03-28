import re
from bs4 import BeautifulSoup


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=' ')
    stripped_text = re.sub(r'[\r\n]+', ' ', stripped_text)
    return stripped_text