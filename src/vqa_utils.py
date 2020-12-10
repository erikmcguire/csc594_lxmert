# coding=utf-8
# Copyleft 2019 Project LXRT

# Modified from above to use iterators from compressed files and to write data.

from io import TextIOWrapper
from zipfile import ZipFile
import pandas as pd
import numpy as np
import sys, csv
import base64
import json
import time
import re

from vqa_paths import DATA_PTH
from vqa_hat import debug

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

SPLIT2FILE = {'train': 'train2014_obj36',
              'trainval': 'val2014_obj36',
              'valid': 'val2014_obj36',
              'test': 'test2015_obj36',
              'mscoco_train': 'mscoco_imgfeat',
              'mscoco_trainval': 'mscoco_imgfeat',
              'mscoco_val': 'mscoco_imgfeat',
              'mscoco_test': 'mscoco_imgfeat'}

SPLIT2JSON = {'mscoco_trainval': 'nominival',
              "mscoco_val": "minival",
              "mscoco_test": "nominival",
              "mscoco_train": "train",
              "train": "train",
              "trainval": "nominival",
              "valid": "minival",
              "test": "test"}

def get_qid2idx():
    trn_q_ids = f"{DATA_PTH}v2_OpenEnded_mscoco_train2014_questions.json"
    val_q_ids = f"{DATA_PTH}v2_OpenEnded_mscoco_val2014_questions.json"
    test_q_ids = f"{DATA_PTH}v2_OpenEnded_mscoco_test2015_questions.json"
    testd_q_ids = f"{DATA_PTH}v2_OpenEnded_mscoco_test-dev2015_questions.json"

    with open(trn_q_ids, 'r') as j: # len: 443,757
         qjs = json.loads(j.read())
    with open(val_q_ids, 'r') as jv: # len: 214,354
         val_qjs = json.loads(jv.read())

    trainval = qjs['questions'] + val_qjs['questions'] # len: 658,111

    with open(test_q_ids, 'r') as jt: # len: 447,793
         test_qjs = json.loads(jt.read())
    with open(testd_q_ids, 'r') as jdt: # len: 107,394
         testdev_qjs = json.loads(jdt.read())

    # {458752000: 0, ... } For indexing into HLAT h5.
    qid2idx = {'trainval': {datum['question_id']: idx
                            for (idx, datum) in enumerate(trainval)},
               'test': {datum['question_id']: idx
                        for (idx, datum) in enumerate(test_qjs['questions'])},
               'test-dev': {datum['question_id']: idx
                           for (idx, datum) in enumerate(testdev_qjs['questions'])}
              }

    return qid2idx

qid2idx = get_qid2idx()

def split2tsv(split):
    if 'val' in split:
        tsv = 'val2014'
    elif 'train' in split:
        tsv = 'train2014'
    elif 'test' in split:
        tsv = 'test2015'
    return tsv

def load_jsn(fn: str):
    print(f"Loading {fn} json.")
    js = json.load(open(f"{DATA_PTH}{fn}.json"))
    return js

def get_js(split):
    js = load_jsn(SPLIT2JSON[split])
    if split != 'test':
        js = filter(lambda x: len(x['label'].values()) != 0, js)
    return js

def load_data(name: str, split: str, chunkSize: int):
    """Load data from tsv or from archive."""
    if 'tsv' in name:
        return load_obj_tsv(name, chunkSize)
    else:
        # Use split to select correct .tsv
        # if multiple files in zip.
        return load_zip(name, split, chunkSize)

def load_zip(zipname: str, split: str, chunkSize: int):
    """Load split's .tsv image data from archive.

       Loads multiple .tsv contained in single zip (mscoco),
       Or single .tsv in single .zip (_obj36).
       Loads chunks as generator to reduce memory.
       Decode, process chunk image data shapes, types.
    """
    with ZipFile(zipname) as zf:
        for fname in zf.infolist():
            zipfilename = fname.filename
            if split in zipfilename:
                with zf.open(fname) as f:
                    print(f'Loading chunk generator from: {zipfilename}')
                    f = TextIOWrapper(f, encoding="utf-8")
                    reader = pd.read_csv(f,
                                         header=None,
                                         names=FIELDNAMES,
                                         sep="\t",
                                         chunksize=chunkSize,
                                         memory_map=True)
                    l = lambda b: np.frombuffer(base64.b64decode(b),
                                                dtype=dtype).reshape(shape)

                    for i, chunk in enumerate(reader):
                        boxes = chunk['num_boxes'].iloc[0]
                        decode_config = [
                            ('objects_id', (boxes, ), np.int64),
                            ('objects_conf', (boxes, ), np.float32),
                            ('attrs_id', (boxes, ), np.int64),
                            ('attrs_conf', (boxes, ), np.float32),
                            ('boxes', (boxes, 4), np.float32),
                            ('features', (boxes, -1), np.float32)
                          ]
                        for key, shape, dtype in decode_config:
                            chunk[key] = chunk[key].iloc[:].map(l)
                        yield chunk

def load_obj_tsv(fname, chunkSize=100):
    """Load object features from tsv file.

       Loads chunks as generator to reduce memory.
       Decode, process chunk image data shapes, types.
    """
    reader = pd.read_csv(fname,
                         header=None,
                         names=FIELDNAMES,
                         sep="\t",
                         chunksize=chunkSize,
                         memory_map=True)
    l = lambda b: np.frombuffer(base64.b64decode(b),
                                dtype=dtype).reshape(shape)

    for i, chunk in enumerate(reader):
        boxes = chunk['num_boxes'].iloc[0]
        decode_config = [
            ('objects_id', (boxes, ), np.int64),
            ('objects_conf', (boxes, ), np.float32),
            ('attrs_id', (boxes, ), np.int64),
            ('attrs_conf', (boxes, ), np.float32),
            ('boxes', (boxes, 4), np.float32),
            ('features', (boxes, -1), np.float32)
          ]
        for key, shape, dtype in decode_config:
            # For toy datasets, some .tsv files already decoded.
            try:
                chunk[key] = chunk[key].iloc[:].map(l)
            except:
                pass
        yield chunk

def write_tsv(fname, chunk_gen):
    """Write/append split's object features to .tsv."""
    start_time = time.time()
    print(f"Writing image data to {fname}.")
    df = pd.DataFrame()
    with open(fname, 'a', newline='') as f:
        writer = csv.DictWriter(f, FIELDNAMES)
        for i, chunk in enumerate(chunk_gen):
            df = pd.concat([df,
                            pd.DataFrame.to_dict(chunk,
                                                 orient='records')])
            writer.writerows(df)
    elapsed_time = time.time() - start_time
    print("Wrote %d images in file %s in %d seconds."
          % (len(df), fname, elapsed_time))
