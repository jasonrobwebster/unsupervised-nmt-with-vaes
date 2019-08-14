'''
Provides helper functions when working with the text data
'''

from __future__ import absolute_import, print_function, unicode_literals

import os
import sys
from random import shuffle

import numpy as np


def all_dataset_types(data_dir='./data/raw/'):
    '''
    Returns a list of available types. Assumes the folder contains the 
    structure:

    ```
    data_dir
    +-- parallel (type_1)
    |   +-- ..
    +-- monolingual (type_2)
    |   +-- ...
    +-- ...
    ```
    Params
    ------
    data_dir: string, default='./data'
        Path of the data folder.

    Returns
    -------

    types: list
        List of all available dataset types
    '''
    return os.listdir(data_dir)


def all_dataset_sources(data_dir='./data/raw/', types='all'):
    '''
    Returns a list of available dataset sources. Assumes the folder contains the 
    structure:

    ```
    data_dir
    +-- parallel (type_1)
    |   +-- source_1
    |   |   +-- ...
    |   +-- source_2
    |   |   +-- ...
    +-- monolingual (type_2)
    |   +-- source_1
    |   |   +-- ...
    |   +-- source_2
    |   |   +-- ...
    ```

    Params
    ------
    data_dir: string, default='./data'
        Path of the data folder.
    
    types: string or list of strings, default='all'
        Type of dataset, either parallel, monolingual, or all.

    Returns
    -------

    sources: list
        List of sources
    '''
    sources = []

    if types == 'all':
        types = all_dataset_types(data_dir=data_dir)
    for t in os.listdir(data_dir):
        if t not in types:
            continue
        type_dir = os.path.join(data_dir, t)
        for s in os.listdir(type_dir):
            sources.append(s)

    return sources
    

def available_datasets(data_dir='./data/raw/', types='all', sources='all'):
    '''
    Returns a list of available datasets. Assumes the folder contains the 
    structure:

    ```
    data_dir
    +-- parallel (type_1)
    |   +-- source_1
    |   |   +-- en_de
    |   |   |   +-- train.en
    |   |   |   +-- train.de
    |   |   |   +-- ...
    |   |   +-- ...
    |   +-- source_2
    |   |   +-- ...
    +-- monolingual (type_2)
    |   +-- source_1
    |   |   +-- en
    |   |   |   +-- mono.en
    |   |   |   +-- ...
    |   |   +-- de
    |   |   |   +-- mono.de
    |   |   |   +-- ...
    |   +-- source_2
    |   |   +-- ...
    ```
    Params
    ------
    data_dir: string, default='./data'
        Path of the data folder.
    
    types: string or list of strings, default='all'
        Type of dataset, either parallel, monolingual, or all.
    
    sources: string or list of strings, default='all'
        Source of the data, such as 'wmt' or 'autshumato'.

    Returns
    -------

    paths: list
        List of directories in the form of ['data_dir/type/source/corpus',...]
    '''

    paths = []

    if types == 'all':
        types = all_dataset_types(data_dir=data_dir)

    if sources == 'all':
        sources = all_dataset_sources()
    
    for t in os.listdir(data_dir):
        if t not in types:
            continue
        type_dir = os.path.join(data_dir, t)
        for s in os.listdir(type_dir):
            if s not in sources:
                continue
            source_dir = os.path.join(type_dir, s)
            for lan in os.listdir(source_dir):
                paths.append(os.path.join(source_dir, lan).replace('\\', '/'))

    return paths


def load_data(path, subword='bpe', vocab_size=30000, format='id', vocab_thresh=50):
    """
    Loads the specified from a given path. The path should be formated as:
    <data/type/source/lang_pair/process>. Presumes the data has been 
    processed by sentencepiece [1].

    Params
    ------

    path: string
        Path to processed files.

    subword: string, default = 'bpe'
        The subword method, e.g. bpe or unigram.

    vocab_size: int, default = 30000
        Size of the vocabulary.

    format: string, default = 'id'
        Should be in ['id', 'seg']. Whether to load the data in a numerical id
        format, or in as a segmented text file.

    vocab_thresh: int, default=50
        Ignores all words in the vocab with occurences below this threshold.

    Returns
    -------

    lang_data: dict
        A dict containing a key value pair {language: (train, test, vocab)} for 
        each language detected in the given path. Vocab data is given as a dict, 
        train/test are returned as a list of lines in string format.

    References:
    -----------
    
    [1] Google Sentencepiece: https://github.com/google/sentencepiece
    """
    assert os.path.exists(path), \
        "Given path {} does not exist".format(path)

    files = os.listdir(path)

    lang_data = dict()

    for f in files:
        names = f.split('.')

        # valid filenames have the form 'set.method.size.format.lang'
        # skip files that are not of this form
        if len(names) < 4:
            continue
        
        lang = names[-1] # file language
        if names[-1] not in lang_data.keys():
            # initizialize data
            lang_data[lang] = [[], [], dict()]
        
        if names[1] == subword and names[2] == str(vocab_size):
            fp = open(os.path.join(path, f), 'r', encoding='utf-8')
            if names[3] == format and (names[0] == 'train' or names[0] == 'test'):
                idx = int(names[0] == 'test') # 0 if train, 1 if test
                data = fp.read().splitlines()
                if format == 'id':
                    data = [np.array(line.split(), dtype=int) for line in data]
                else:
                    data = [np.array(line.split()) for line in data]
                lang_data[lang][idx].extend(data)
            elif names[0] == 'vocab':
                for line in fp:
                    key, val = line.split()
                    vocab = lang_data[lang][2]
                    if int(val) >= vocab_thresh:
                        vocab[key] = int(val)
            fp.close()
    
    return lang_data

def load_vocab(path, subword='bpe', vocab_size=30000):

    assert os.path.exists(path), \
        "Given path {} does not exist".format(path)

    files = os.listdir(path)

    joint_vocab = dict()
    joint_vocab_path = '{}.{}.vocab'.format(subword, str(vocab_size))
    if joint_vocab_path in files:
        fp = open(os.path.join(path, joint_vocab_path), 'rt', encoding='utf-8')
        for i, line in enumerate(fp):
            data = line.split()
            key = data[0]
            joint_vocab[key] = i
        fp.close()

        if len(joint_vocab) == 0:
            joint_vocab = None
        return joint_vocab


def process_and_mask_ids(
        data, 
        seq_len=256, 
        padding='left', 
        keep='start', 
        start_token=1, 
        end_token=2
    ):
    """
    Takes data that contains a list of lines in string form, and returns a
    numpy array of shape [nb_sentences, seq_len], as well as a mask for the
    padding sequence.

    Params
    ------

    data: list
        List of raw sentences containing IDs

    seq_len: int,  default = 256
        The length of the processed sequence. 

    padding: string, default = 'left'
        Either 'left' or 'right' padding. Fills a sequences to the final
        sequence length.

    keep: string, default = 'start'
        Either 'start' or 'end'. Whether to keep the start or the end of a
        sequence, if the sequence is longer than the desired sequence length.

    start_token: int, default = 1
        The token used to indicate that a sentence has begun.

    end_token: int, default = 2
        The used to indicate a sentence has ended.

    Returns
    -------

    processed: array
        The processed data, of shape S x L, where S is the number of sentences
        in the given data, and L is the desired sequence length.
    """
    assert seq_len > 0, \
        "Sequence length must be greater than 0."

    assert padding in ['left', 'right'], \
        "Padding should either be 'left' or 'right'."

    assert keep in ['start', 'end'], \
        "You should choose to keep either the start or end of a sentence."

    processed = np.zeros((len(data), seq_len), dtype=int)
    mask = np.zeros((len(data), seq_len), dtype=int)

    for i, line in enumerate(data):
        line = np.array(line.split(), dtype=int)
        if len(line) > seq_len-2:
            if keep == 'start':
                line = line[:seq_len-2]
            else:
                line = line[-seq_len+2:]
        line = np.concatenate([[start_token], line, [end_token]])
        if padding == 'left':
            processed[i, -len(line):] = line
            mask[i, -len(line):] = 1
        else:
            processed[i, :len(line)] = line
            mask[i, :len(line)] = 1
    
    return processed, mask


def batch_process_and_mask_ids(
        data,
        batch_size=32,
        shuffle=True,
        seq_len=256, 
        padding='left', 
        keep='start', 
        start_token=1, 
        end_token=2):
    """
    Takes data that contains a list of lines in string form, and returns a
    generator producing a numpy array of shape [batch_size, seq_len], 
    as well as a mask for the padding sequence. 

    Params
    ------

    data: list
        List of raw sentences containing IDs

    batch_size: int, default = 32
        The batch size.

    shuffle: bool, default = True
        Whether to shuffle the data over each epoch.

    seq_len: int,  default = 256
        The length of the processed sequence. 

    padding: string, default = 'left'
        Either 'left' or 'right' padding. Fills a sequences to the final
        sequence length.

    keep: string, default = 'start'
        Either 'start' or 'end'. Whether to keep the start or the end of a
        sequence, if the sequence is longer than the desired sequence length.

    start_token: int, default = 1
        The token used to indicate that a sentence has begun.

    end_token: int, default = 2
        The used to indicate a sentence has ended.

    Returns
    -------

    processed: array
        The processed data, of shape S x L, where S is the number of sentences
        in the given data, and L is the desired sequence length.
    """
    if shuffle:
        shuffle(data)

    steps_per_epoch = np.ceil(len(data) / batch_size)
    step = 0

    while True:
        processed, mask = process_and_mask_ids(
            data[step*batch_size:(step+1)*batch_size],
            seq_len=seq_len,
            padding=padding,
            keep=keep,
            start_token=start_token,
            end_token=end_token
        )
        step += 1
        if step > steps_per_epoch:
            shuffle(data)
            step = 0
        yield processed, mask


if __name__ == '__main__':
    data = [
        '6 7 8 9 0 6 7 8 9',
        '9 8 7 6 5 0 9 7 6 5',
        '2 3 4 5 6 7 8 9 10 11 12 13 14 15 16'
        ]
    processed, mask = process_and_mask_ids(
        data, seq_len=15, padding='right', keep='start'
    )
    print(processed)
    print(mask)