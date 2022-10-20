# python dependency_display.py sentence
import sys
from spacy import displacy, load
nlp = load('en_core_web_lg')

doc = nlp(sys.argv[1])
displacy.serve(doc, style='dep', options={'compact':False}, port=5001)