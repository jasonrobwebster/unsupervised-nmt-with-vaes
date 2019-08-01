'''
Provides helper functions when working with the data
'''

from __future__ import absolute_import, print_function, unicode_literals

import os
import sys


def all_dataset_types(data_dir='./data/'):
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


def all_dataset_sources(data_dir='./data/', types='all'):
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
    

def available_datasets(data_dir='./data/', types='all', sources='all'):
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

if __name__ == '__main__':
    print(available_datasets())