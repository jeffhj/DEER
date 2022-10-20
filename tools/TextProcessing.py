import json
from tools.BasicUtils import my_write, MyMultiProcessing, calculate_time
import numpy as np
import re
import tqdm
from typing import Dict
import pickle
from nltk import WordNetLemmatizer, pos_tag, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en_core_web_lg')

wnl = WordNetLemmatizer()

filtered_words = set(['can', 'it', 'work', 'in', 'form', 'parts', 'is', 'its', 'or', 'and', 'a','b','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', ''])
filtered_words.update(set(stopwords.words('english')))

def filter_specific_keywords(keywords:list):
    return [kw for kw in keywords if kw not in filtered_words]

def process_keywords(keywords:list):
    keywords = filter_specific_keywords(keywords)
    stable_kw = []
    unstable_kw = []
    for kw in keywords:
        # if '- ' in kw:
        #     continue
        # splited = kw.replace('-', ' - ')
        # reformed = ' '.join(sent_lemmatize(splited))
        reformed = ' '.join(sent_lemmatize(kw))
        # if reformed == splited:
        if reformed == kw:
            stable_kw.append(kw)
        else:
            unstable_kw.append('%s\t%s' % (kw, reformed))
    return stable_kw, unstable_kw

def normalize_text(text:str):
    tokens = text.lower().split()
    refine_tokens = []
    for token in tokens:
        if token.isalnum():
            refine_tokens.append(token)
        else:
            temp_str = ''
            for char in token:
                if char.isalnum() or char == "'" or char == '.':
                    temp_str += char
                else:
                    if temp_str:
                        refine_tokens.append(temp_str)
                    refine_tokens.append(char)
                    temp_str = ''
            if temp_str:
                refine_tokens.append(temp_str)
    return ' '.join(refine_tokens)

def remove_brackets(text:str):
    while re.search(r'{[^{}]*}', text):
        text = re.sub(r'{[^{}]*}', '', text)
    while re.search(r'\([^()]*\)', text):
        text = re.sub(r'\([^()]*\)', '', text)
    while re.search(r'\[[^][]*\]', text):
        text = re.sub(r'\[[^][]*\]', '', text)
    return ' '.join(text.split())

def clean_text(text:str):
    return ' '.join(re.sub(r'[^a-z0-9,.;\s-]', '', remove_brackets(normalize_text(text))).split())

@calculate_time
def build_word_tree(input_txt:str, dump_file:str, entity_file:str):
    MyTree = {}
    entities = []
    cnt = 0
    with open(input_txt, 'r', encoding='utf-8') as load_file:
        keywords = load_file.readlines()
        for word in tqdm.tqdm(keywords):
            # Directly add the '_' connected keyword
            word = word.strip()
            phrase = word.split()
            if not phrase:
                continue
            entities.append('_'.join(phrase))
            cnt += 1
            # Insert the keyword to the tree structure
            if len(phrase) == 1:
                # If the word is an atomic word instead of a phrase
                if word not in MyTree.keys():
                    # If this is the first time that this word is inserted to the tree
                    MyTree[word] = {"":""}
                elif "" not in MyTree[word].keys():
                    # If the word has been inserted but is viewed as an atomic word the first time
                    MyTree[word][""] = ""
                # If the word has already been inserted as an atomic word, then we do nothing
            else:
                # If the word is an phrase
                length = len(phrase)
                fw = phrase[0]
                if fw not in MyTree.keys():
                    MyTree[fw] = {}
                temp_dict = MyTree.copy()
                parent_node = fw
                for i in range(1, length):
                    if phrase[i]:
                        sw = phrase[i]
                        if sw not in temp_dict[parent_node].keys():
                            # The second word is inserted to as the child of parent node the first time
                            temp_dict[parent_node][sw] = {}
                        if i == length - 1:
                            # If the second word is the last word in the phrase
                            if "" not in temp_dict[parent_node][sw].keys():
                                temp_dict[parent_node][sw][""] = ""
                        else:
                            # If the second word is not the last word in the phrase
                            temp_dict = temp_dict[parent_node].copy()
                            parent_node = sw
        print('Building word tree is accomplished with %d words added' % (cnt))
    with open(dump_file, 'w', encoding='utf-8') as output_file:
        json.dump(MyTree, output_file)
    my_write(entity_file, entities)


def build_word_tree_v2(input_list, dump_file:str='', token_file:str='', old_MyTree:dict=None, old_token2idx:dict=None):
    if old_MyTree is not None and old_token2idx is not None:
        MyTree = old_MyTree
        token2idx = old_token2idx
        token_idx = max([v for k, v in token2idx.items()]) + 1
    else:
        MyTree = {}
        token2idx = {}
        token_idx = 0
        
    if type(input_list) == str:
        with open(input_list, 'r', encoding='utf-8') as load_file:
            keywords_str = load_file.read().strip()
            # transform keywords into index
            keywords = keywords_str.splitlines()
    elif type(input_list) == list:
        keywords = input_list
    else:
        return
    
    for kw in keywords:
        for token in kw.split():
            if token not in token2idx:
                token2idx[token] = token_idx
                token_idx += 1
    keywords_idx = [[token2idx[token] for token in kw.split()] for kw in keywords]
    # start building wordtree
    for word in keywords_idx:
        if not word:
            print('Bad word in', keywords)
            return
        # Insert the keyword to the tree structure
        if len(word) == 1:
            # If the word is an atomic word instead of a phrase
            if word[0] not in MyTree:
                # If this is the first time that this word is inserted to the tree
                MyTree[word[0]] = {-1:-1}
            elif -1 not in MyTree[word[0]]:
                # If the word has been inserted but is viewed as an atomic word the first time
                MyTree[word[0]][-1] = -1
            # If the word has already been inserted as an atomic word, then we do nothing
        else:
            # If the word is an phrase
            length = len(word)
            fw = word[0]
            if fw not in MyTree:
                MyTree[fw] = {}
            temp_dict = MyTree
            parent_node = fw
            for i in range(1, length):
                sw = word[i]
                if sw not in temp_dict[parent_node]:
                    # The second word is inserted to as the child of parent node the first time
                    temp_dict[parent_node][sw] = {}
                if i == length - 1:
                    # If the second word is the last word in the phrase
                    if -1 not in temp_dict[parent_node][sw]:
                        temp_dict[parent_node][sw][-1] = -1
                else:
                    # If the second word is not the last word in the phrase
                    temp_dict = temp_dict[parent_node]
                    parent_node = sw
    if dump_file:
        print('Building word tree is accomplished with %d words added' % (len(keywords)))
        with open(dump_file, 'wb') as output_file:
            pickle.dump(MyTree, output_file)
    if token_file:
        with open(token_file, 'wb') as output_file:
            pickle.dump(token2idx, output_file)
        
    return MyTree, token2idx


def sent_lemmatize(sentence):
    if type(sentence) == str:
        sentence = word_tokenize(sentence)
    return [str(wnl.lemmatize(word, pos='n') if tag.startswith('NN') else word) for word, tag in pos_tag(sentence)]

def batched_sent_tokenize(paragraphs:list):
    content = []
    for para in paragraphs:
        content += sent_tokenize(para)
    return content

def find_root_in_span(kw:spacy.tokens.span.Span)->int:
    span_front = kw[0].i
    span_back = kw[-1].i
    root = kw[-1]
    while root.head.i >= span_front and root.head.i <= span_back and root.head.i != root.i:
        root = root.head
    return root.i

def find_dependency_path_from_tree(doc, kw1:spacy.tokens.span.Span, kw2:spacy.tokens.span.Span):
    # Find roots of the spans
    idx1 = find_root_in_span(kw1)
    idx2 = find_root_in_span(kw2)
    branch = np.zeros(len(doc))
    i = idx1
    while branch[i] == 0:
        branch[i] = 1
        i = doc[i].head.i
    i = idx2
    while branch[i] != 1:
        branch[i] = 2
        if i == doc[i].head.i:
            return ''
        i = doc[i].head.i
    dep1 = []
    j = idx1
    while j != i:
        dep1.append('i_%s' % doc[j].dep_)
        j = doc[j].head.i
    dep2 = []
    j = idx2
    while j != i:
        dep2.append(doc[j].dep_)
        j = doc[j].head.i
    dep2.reverse()
    if branch[idx2] == 1:
        # kw2 is along the heads of kw1
        return ' '.join(dep1)
    elif i == idx1:
        # kw1 is along the heads of kw2
        return ' '.join(dep2)
    else:
        return ' '.join(dep1 + dep2)


def find_span(doc:spacy.tokens.doc.Doc, phrase:str, use_lemma:bool=False, lower:bool=False):
    """
    Find all the occurances of a given phrase in the sentence using spacy.tokens.doc.Doc

    Inputs
    ----------
    doc : spacy.tokens.doc.Doc
        a doc analyzed by a spacy model

    phrase : str
        the phrase to be searched

    use_lemma : bool
        if true, the lemmazation is applied to each token before searching

    Return
    -------
    A list of phrases (spacy.tokens.span.Span) found in the doc
    """
    nltk_word_tokenized = word_tokenize(doc.text)
    nltk_word_tokenized_normalized = ['"' if t == '``' or t == '\'\'' else t for t in nltk_word_tokenized]
    spacy_word_tokenized = [t.text for t in doc]
    use_spacy = nltk_word_tokenized_normalized != spacy_word_tokenized
        
    if use_spacy:
        phrase_doc = nlp(phrase)
        phrase_tokens = [str(t.lemma_ if use_lemma and t.pos_ == 'NOUN' else t) for t in phrase_doc]
        sent_tokens = [str(t.lemma_ if use_lemma and t.pos_ == 'NOUN' else t) for t in doc]
    else:
        phrase_tokens = phrase.replace('-', ' - ').split()
        sent_tokens = sent_lemmatize(nltk_word_tokenized)

    phrase_length = len(phrase_tokens)
    if lower:
        sent_tokens = [token.lower() for token in sent_tokens]
        phrase_tokens = [token.lower() for token in phrase_tokens]
    
    return [doc[i : i + phrase_length] for i in range(len(doc)-phrase_length+1) if phrase_tokens == sent_tokens[i : i + phrase_length]]


def find_noun_phrases(doc:spacy.tokens.doc.Doc):
    """
    Find all the noun phrases in the sentence using spacy.tokens.doc.Doc

    Inputs
    ----------
    doc : spacy.tokens.doc.Doc
        a doc analyzed by a spacy model

    Return
    -------
    A list of noun phrases (spacy.tokens.span.Span) collected from the doc
    """
    noun_phrases_list = [s for s in doc.noun_chunks if s[-1].pos_ != 'PRON']
    if len(noun_phrases_list) < 2:
        return noun_phrases_list
    merge_noun_phrases = [noun_phrases_list[0]]
    for i in range(1, len(noun_phrases_list)):
        if noun_phrases_list[i-1][-1].i == (noun_phrases_list[i][0].i - 1):
            merge_noun_phrases[-1] = doc[noun_phrases_list[i-1][0].i : noun_phrases_list[i][-1].i+1]
        else:
            merge_noun_phrases.append(noun_phrases_list[i])
    return merge_noun_phrases


def exact_match(pattern:str, string:str):
    """
    Check whether the string exactly matches the re.Pattern. "Exactly" here means no extra substr is out of the pattern.

    Inputs
    ----------
    pattern : re.Pattern
        the re pattern

    string : str
        the string to be examined

    Return
    -------
    True if the string matches the pattern exactly
    """
    mat = re.compile(pattern).match(string)
    if mat is None:
        return False
    return len(string) == mat.end()


def my_sentence_tokenize(paragraph:str, use_spacy:bool=False):
    """
    Tokenize a paragraph string to sentences.

    Inputs
    ----------
    paragraph : str
        the paragraph string

    use_spacy : bool
        use spacy if true, nltk sent_tokenize otherwise.

    Return
    -------
    List of sentences in the paragraph
    """
    if use_spacy:
        return [str(s) for s in nlp(paragraph).sents]
    else:
        return sent_tokenize(paragraph)