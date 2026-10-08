# Copyright (c) 2017 Keith Ito
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" from https://github.com/keithito/tacotron 
Modified to add puncturation removal
"""

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
    1. "english_cleaners" for English text
    2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
         the Unidecode library (https://pypi.python.org/pypi/Unidecode)
    3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
         the symbols in symbols.py to match your data).

'''

import re
from unidecode import unidecode
from .numbers import normalize_numbers

#Speedyloader
import time

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

def expand_abbreviations(text):
    #print("\nSpeedyloader: expand_abbreviations")
    start = time.time()
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    #print("\nSpeedyloader-expand_abreviation: time=", (time.time()-start)*1000000, "micros, len=", len(text))
    return text

def expand_numbers(text):
    #print("\nSpeedyloader: expand_numbers")
    start = time.time()
    tmp = normalize_numbers(text)
    #print("\nSpeedyloader-expand_numbers: time=", (time.time()-start)*1000000, "micros, len=", len(text))
    return tmp #normalize_numbers(text)

def lowercase(text):
#    print("\nSpeedyloader: lowercase")
    start = time.time()
    tmp = text.lower()
    #print("\nSpeedyloader-lowercase: time=", (time.time()-start)*1000000, "micros, len=", len(text))
    return tmp #text.lower()

def collapse_whitespace(text):
    start = time.time()
    tmp = re.sub(_whitespace_re, ' ', text)
    #print("\nSpeedyloader-collapse_whitespace: time=", (time.time()-start)*1000000, "micros, len=", len(text))
    return tmp #re.sub(_whitespace_re, ' ', text)

def convert_to_ascii(text):
    start = time.time()
    tmp = unidecode(text)
    #print("\nSpeedyloader-collapse_convert_to_ascii: time=", (time.time()-start)*1000000, "micros, len=", len(text))
    return tmp #unidecode(text)

def remove_punctuation(text, table):
    start = time.time()
    text = text.translate(table)
    text = re.sub(r'&', " and ", text)
    text = re.sub(r'\+', " plus ", text)
    #print("\nSpeedyloader-remove_punctuation.text : time=", (time.time()-start)*1000000, "micros, len=", len(text))
    return text

def english_cleaners(text, table=None):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    start = time.time()
    #print("\nSpeedyloader: english_cleaners")
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    if table is not None:
        text = remove_punctuation(text, table)
    text = collapse_whitespace(text)
    #print("\nSpeedyloader-english_cleaners- time=", (time.time()-start)*1000000, "micros, len=", len(text))
    return text
