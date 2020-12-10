# coding=utf-8
# Copyleft 2019 project LXRT.

# Modified from above to use Dataframe chunks, VQA-HLAT, HuggingFace Trainer.

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import json
import os

from vqa_paths import LABEL_PTH, VQA_PTH, MSCOCO_IMGFEAT_ROOT
from vqa_utils import *
from vqa_param import args
from vqa_hat import *

class VQADataset:
    """Collate QA data from json files."""

    def __init__(self, splits: str):

        # {word: index}
        self.ans2label = json.load(open(LABEL_PTH))

        # [words]
        self.label2ans = json.load(open(VQA_PTH))

        assert len(self.ans2label) == len(self.label2ans)

        # Training uses train, validation (nominival) splits' JSON data.
        self.splits = splits.split(',') # e.g. args.train: "train,nominival"
        self.json_data = []
        for split in self.splits:
            # [train.json, nominival.json], [minival.json], etc.
            self.json_data.extend(get_js(split))

        # Exclude training, validation data without labels.
        self.json_data = [d if "label" in d.keys() and d["label"] != {} and
                            'test' not in splits else d for d in self.json_data]

    @property
    def num_answers(self):
        """Total answers in training, validation data."""
        return len(self.ans2label)

    @property
    def data(self):
        """QA dictionaries from JSON file(s)."""
        return self.json_data

    def __len__(self):
        """Number of QA dictionaries."""
        return len(self.data)

class VQATorchDataset(Dataset):
    """Match image information to JSON data.

       Use generators due to image feature size.
       Provide encoded, QA pairs matched with
       image features from .tsv files.

       Optionally, process and match human attentions
       for pairs.
    """

    def __init__(self, dataset: VQADataset, tokenizer, is_valid):
        super().__init__()

        self.tokenizer = tokenizer

        self.id2im = pd.DataFrame() # {img_id: img_data}

        # Human attentions if training or saving
        self.h_att_all = get_h_att() if not is_valid else None

        chunk_generators = []
        # Training Dataset uses img data chunks from train & nominival splits.
        for split in dataset.splits:
            spth = os.path.join(MSCOCO_IMGFEAT_ROOT,
                               f'{SPLIT2FILE[split]}.{args.ext}')
            chunk_generator = load_data(spth,
                              split=split2tsv(split),
                              chunkSize=args.chunkSize)
            chunk_generators.append(chunk_generator)
            print(f"Got chunk generator for {split}.")
            if args.write == 'write':
                wpth = os.path.join(MSCOCO_IMGFEAT_ROOT,
                                    f'{SPLIT2FILE[split]}.tsv')
                write_tsv(wpth, chunk_generator)

        # Match image data from chunks with json data.
        # If more than one chunk generator ([train, nominival]),
        # collate matching image data from chunks in both.
        self.json_set = dataset # QA data from JSON
        self.data = [] # QA data matched to image data
        jd = pd.DataFrame.from_dict(self.json_set.data)

        # Iterate train+val (or minival or test) set's chunk generator(s).
        for j, chunk_generator in enumerate(chunk_generators):
            # Iterate img data chunks in chunk generator.
            for i, cnk in enumerate(chunk_generator):
                # Limit to save memory.
                if args.topk and len(self.id2im) >= args.topk:
                    break

                if i > 0: # Once we have image data stored.

                    # Matched img-QA data based on shared image ids.
                    inc = jd[jd.img_id.isin(self.id2im.img_id)]

                    # Reduce search space/RAM by setting to remainder.
                    jd = jd[~jd.img_id.isin(self.id2im.img_id)]

                    # Store matches' QA JSON data.
                    self.data.extend(pd.DataFrame.to_dict(inc,
                                                          orient='records'))
                # Store image data from chunk.
                cnk_match = cnk[cnk.img_id.isin(jd.img_id)]
                self.id2im = pd.concat([self.id2im, cnk_match])

        sa = f"\nUse {len(self.data)} data in torch"
        sb = f" dataset from {len(self.id2im)} images.\n"
        print(sa + sb)

        # Encode matched questions so datum indices match in __getitem__.
        self.encodings = self.tokenize()

    def questions(self):
        """List of questions from collated dictionaries."""
        return [d["sent"] for d in self.data]

    def __len__(self):
        """Number of image-QA pairs."""
        return len(self.data)

    def tokenize(self):
        """Encode questions from collated dictionaries."""
        return self.tokenizer(self.questions(),
                              truncation=True,
                              padding="max_length",
                              max_length=30,
                              return_token_type_ids=True,
                              return_attention_mask=True,
                              add_special_tokens=True,
                              return_tensors="pt")

    def __getitem__(self, idx: int):
        # Get item's tokenizer data (ids, mask, etc.).
        item = {key: val[idx]
                for key, val in self.encodings.items()}

        # Get item's matched JSON data.
        datum = self.data[idx]

        # To export unlabeled test preds to .json in custom evaluate().
        ques_id = datum['question_id']
        item["question_id"] = int(ques_id)

        # Image associated w/ item's matched JSON data.
        img_info = self.id2im[self.id2im['img_id'] == datum['img_id']]

        # Visual features
        visual_feats = img_info['features'].values[0]

        # Spatial features
        obj_num = img_info['num_boxes'].values[0]
        visual_pos = img_info['boxes'].values[0]

        assert obj_num == len(visual_pos) == len(visual_feats)

        img_h, img_w = (img_info['img_h'].values[0],
                        img_info['img_w'].values[0])

        # Obtain human data for attention supervision and/or logging.
        # Alternatively to conditional here, may exclude in subclassed
        # prediction_step to avoid model .forward() errors.
        if self.h_att_all:
            # Search for matching human attentions
            # in each split as necessary.
            for split in ['trainval', 'test-dev', 'test']:
                try:
                    idx = qid2idx[split][ques_id]
                    hdatum = self.h_att_all[split][idx]
                    break
                except:
                    pass

            # From flattened maps used in original HAN.
            hdatum = np.reshape(hdatum, (14, 14))

            # Reshape, extract from boxes, re-normalize.
            hdatum = get_hatt_boxed(visual_pos,
                                    img_h, img_w,
                                    hdatum)

            # Distribution over objects for this img-question pair.
            item["h_att"] = hdatum

        # Normalize bounding boxes (after getting human attentions).
        # x1, y1, x2, y2
        visual_pos = visual_pos.copy()
        visual_pos[:, (0, 2)] /= img_w
        visual_pos[:, (1, 3)] /= img_h
        np.testing.assert_array_less(visual_pos, 1+1e-5)
        np.testing.assert_array_less(-visual_pos, 0+1e-5)

        # Use keys corresponding to expected by model .forward().
        item["visual_feats"] = visual_feats
        item["visual_pos"] = visual_pos

        # HF modeling uses scalar labels; e.g. batch shape: (batch_size).
        # datum['label'] (a dict) may contain
        # multiple (answer: score) pairs.
        if 'label' in datum:
            label = datum['label']
            try:
                # Use answer w/ max score (e.g. 1).
                ans = max(label, key=label.get)
            except:
                # Or list() and slice if not a dict.
                ans = list(label.values())[0]
            # Get answer index.
            target = self.json_set.ans2label[ans]
            item["labels"] = target
        return item
