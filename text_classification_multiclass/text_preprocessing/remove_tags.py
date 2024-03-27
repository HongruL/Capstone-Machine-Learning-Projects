import requests, re
from bs4 import BeautifulSoup

data = requests.get('http://www.gutenberg.org/cache/epub/8001/pg8001.html')
content = data.content
def strip_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r\n]+', '\n', stripped_text)
    return stripped_text

clean_content = strip_html_tags(content)
#print(clean_content[1163:2045])