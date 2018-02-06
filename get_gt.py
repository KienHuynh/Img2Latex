"""
Created on Thu Aug  3 21:30:37 2017

This file consists of functions that parse the latex XML tree into a list of latex symbols (string), which can be used as ground truth for inference/training

@author: ngocbui
Modified by Kien Huynh
"""

import sys
import numpy as np
import xml.etree.ElementTree as r
import os
import collections

import pdb

def read_symbol_file(path):
    assert(os.path.exists(path))
    with open(path, 'r') as f:
        return f.read().replace("\n", " ").split()


def build_vocab(path):
    """build_vocab
    :param path: path to the math symbol txt file
    
    :return word_to_id, id_to_word: string to index dictionary and vice versa
    """
    data = read_symbol_file(path)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items()) 
    return word_to_id, id_to_word


def symbol2id(symbol_list, word_to_id):
    """symbol2id
    Convert tex symbols (string) to their corresponding ids (in the word_to_id dict built from build_vocab)

    :param symbol_list: list of latex symbols (string)
    :word_to_id: dictionary, return from build_vocab
    """
    return [word_to_id[symbol] for symbol in symbol_list if symbol in word_to_id]


#def touchGT(path):
#    assert(os.path.exists(path))
#    root = r.parse(path).getroot()
#    #print('parse', r.parse(path))
#    #print(root.tag)
#    #print(root.getchildren())
#    tag_header_len = len(root.tag)-3
#    
#    for child in root:
#        tag = child.tag[tag_header_len:]
#        #print(tag)
#        if tag == 'annotation' and child.attrib['type'] == 'truth':
#            text = child.text
#            #print(text)
#            text = text.replace('$','')
#            #print(text)
#    #        print(text.split())
#    #        text = text.split()
#    return text


def get_root(path):
    """get_root
    Return the root of an XML tree given a path
    
    :param path: full path to the inkml file
    
    :return root: root of the XML tree
    """
    assert(os.path.exists(path))
    root = r.parse(path).getroot()
    return root


def standardize_text(text):
    """modify_text
    Some latex symbols were written in different representations (for example, lt and <) in the inkml files. This function maps all variations of these symbols into one.
    
    :param text: text to be modified
    
    :return standtext: standardized text
    """
    standard = ['phi','pi','theta','alpha','beta','gamma','infty','sigma','Delta',
                'lamda','mu','pm','sin','cos','neq','leq','gt','sqrt','div','times',
                'sum','log','tan','ldots','geq','rightarrow','lim','int','exists',
                'forall','in','prime','lt','ne','cdot','cdots']
    
    if text == '<':
        text = 'lt'
    elif text =='>':
        text = 'gt'
    elif text == 'im':
        text = 'lim'
    elif text == '.':
        text = 'cdot'
    elif text == 'ctdot':
        text = 'cdots'
        
    if text in standard:
        standtext = '\\'+text
    else:
        standtext = text
    return standtext


def get_index(root):
    """get_index
    Return the index of the } symbol + 1
    
    :param root: current node
    """
    index = root.tag.index('}') + 1
    return index


def parse_latex_tree(root, output_text, ignore_elements):
    """parse_latex_tree
    Parse an XML latex tree to a list of latex string. The result will be stored in the param output_text.
    
    :param root: root of the latex XML tree, loaded from an inkml file
    :param output_text: output list containing latex symbols (for example, ['<s>', '\frac', '{2}', '{3}', '</s>'])
    :param ignore_elements: elements not related to latex in the XML tree will be ignored
    """
    index = get_index(root)
    if root.tag[index:] in ignore_elements:
        return
    if len(root) == 0:   
        temp = standardize_text(root.text)
        output_text.append(temp)
        return
    else:
        if root.tag[index:] == 'msqrt':            
            output_text.append('\\sqrt')
            output_text.append('{')
            for child in root:
                parse_latex_tree(child, output_text, ignore_elements)
            output_text.append('}')
        elif root.tag[index:] == 'mfrac':
            output_text.append('\\frac')
            for child in root:
                output_text.append('{')
                parse_latex_tree(child, output_text, ignore_elements)
                output_text.append('}')  
        elif root.tag[index:] == 'msub' or root.tag[index:] == 'munder':
            n = 1
            for child in root:
                if n == 2:
                    output_text.append('_')
                    if child.tag[index:] == 'mrow':
                        output_text.append('{')
                        parse_latex_tree(child, output_text, ignore_elements)
                        output_text.append('}')
                    else:
                        parse_latex_tree(child, output_text, ignore_elements)
                else:
                    parse_latex_tree(child, output_text, ignore_elements)
                n = n + 1
        elif root.tag[index:] == 'msup' or root.tag[index:] == 'mover':
            n = 1
            for child in root:
                if n == 2:
                    output_text.append('^')
                    if child.tag[index:] == 'mrow':
                        output_text.append('{')
                        parse_latex_tree(child, output_text, ignore_elements)
                        output_text.append('}')
                    else:
                        parse_latex_tree(child, output_text, ignore_elements)
                else:
                    parse_latex_tree(child, output_text, ignore_elements)
                n = n + 1
        elif root.tag[index:] == 'msubsup' or root.tag[index:] == 'munderover':
            n = 1
            for child in root:
                if n == 2:
                    output_text.append('_')
                    if child.tag[index:] == 'mrow':
                        output_text.append('{')
                        parse_latex_tree(child, output_text, ignore_elements)
                        output_text.append('}')
                    else:
                        parse_latex_tree(child, output_text, ignore_elements)
                elif n == 3:
                    output_text.append('^')
                    if child.tag[index:] == 'mrow':
                        output_text.append('{')
                        parse_latex_tree(child, output_text, ignore_elements)
                        output_text.append('}')
                    else:
                        parse_latex_tree(child, output_text, ignore_elements)
                else:
                    parse_latex_tree(child, output_text, ignore_elements)
                n = n + 1
            
        else:
            for child in root:
                parse_latex_tree(child, output_text, ignore_elements)
    
    
def read_latex_label(path_to_ink, path_to_symbol, max_len):
    """read_latex_label
    Read latex sequence from inkml file and return the corresponding label vector
    
    :param path_to_ink: path to the inkml file
    :param path_to_symbol: path to the file containing latex symbols to be learnt
    :param max_len: maximum allowed length for a latex sequence. Sequences shorter than this number will ba padded, longer sequences will be cut off)

    :return vector: numpy 1D vector, the word index vector
    """

    word_to_id, id_to_word = build_vocab(path_to_symbol)
    root = get_root(path_to_ink)
    ignore_elements = ['traceFormat','annotation','trace','traceGroup']
    symbol_list = ['<s>']
    parse_latex_tree(root, symbol_list, ignore_elements)
    
    need_to_pad = max_len - len(symbol_list)
    
    if (len(symbol_list) > max_len):
        symbol_list = symbol_list[0:(max_len-1)]
    
    symbol_list.append('</s>')

    for i in range(need_to_pad):
        symbol_list.append('$P')

    vector = symbol2id(symbol_list, word_to_id)
    return vector
