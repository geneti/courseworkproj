import re

def clean(string):
    string = re.sub(r'[^\x00-\x7F]+','', string).lower()
    string = re.sub(r'(\n|\t|\r)+','', string)
    return string
