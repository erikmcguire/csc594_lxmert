# Combination of LXMERT demo notebook and various code examples to visualize human attentions alongside LXMERT.

from typing import Dict, Mapping, List
from scipy.special import softmax

from transformers import (
    LxmertForQuestionAnswering,
    LxmertTokenizer,
    )

from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess
from utils import Config

from bertviz import head_view, model_view
from bertviz.neuron_view import show

import numpy as np
import PIL.Image
import io
import json
import csv
import h5py

from os.path import isfile, join
from os import listdir
import os

import torch
import utils
import pickle

import urllib.request
import cv2

import matplotlib.pyplot as plt
import matplotlib

from PIL import Image as imager

from vqa_paths import (
    HAT_DATA,
    OBJ_PTH,
    ATTR_PTH,
    VQA_PTH,
    LABEL_PTH,
    MODEL_PTH
    )

# load object, attribute, and answer labels
with open(OBJ_PTH) as objf:
    objids = objf.read().splitlines()
with open(ATTR_PTH) as attf:
    attrids = attf.read().splitlines()
vqa_answers = utils.get_data(VQA_PTH)
vqa_labels = utils.get_data(LABEL_PTH)

# load models and model components
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned",
                                        config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)

tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

def get_h_att():
    """
    Load VQA 2.0 HLAT files.
    {"pre_attmap" : attention maps for question ids (ordered)}
    https://github.com/qiaott/HAN
    for question-image pairs:
        train_val: 658,111 attention maps
        test_dev:  107,394 attention maps
        test:      447,793 attention maps
    """
    trainval = h5py.File(f'{HAT_DATA}trainval2014_attention_maps.h5', 'r')
    test_dev = h5py.File(f'{HAT_DATA}test-dev2015_attention_maps.h5', 'r')
    test =         h5py.File(f'{HAT_DATA}test2015_attention_maps.h5', 'r')
    return {'trainval': trainval['pre_attmap'],
            'test-dev': test_dev['pre_attmap'],
                    'test': test['pre_attmap']}

h_att_data = get_h_att()

def get_hatt(id, ip, question_ids, s=None):
    tups = []
    for ix, x in enumerate(question_ids):
        id = str(id)
        if id[0] == "0":
            id = int(id[1:])
        if x["image_id"] == int(id):
            tups.append((ix, x['question']))
    if s:
        for ix, x in enumerate(question_ids):
            if s in x["question"]:
                tups.append((ix, x['question']))
    return tups

def f(img, p):
    ip = p + img
    id = img[-10:-4]
    img = cv2.imread(ip)
    plt.imshow(img)
    plt.show()
    return ip, id

def showarray(s, size=(10, 10)):
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(s)
    plt.tight_layout()

def get_bboxes(ip, frcnn, max_detections=None, viz=False):
    # run frcnn
    images, sizes, scales_yx = image_preprocess(ip)
    if max_detections:
        md = max_detections
    else:
        md = frcnn_cfg.max_detections
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=md,
        return_tensors="pt"
    )
    if viz:
        #image viz
        frcnn_visualizer = SingleImageViz(ip, id2obj=objids, id2attr=attrids)
        # add boxes and labels to the image

        frcnn_visualizer.draw_boxes(
            output_dict.get("boxes"),
            output_dict.pop("obj_ids"),
            output_dict.pop("obj_probs"),
            output_dict.pop("attr_ids"),
            output_dict.pop("attr_probs"),
        )

        j = frcnn_visualizer._get_buffer()
    else:
        j = None
    return output_dict, j

def get_qa(output_dict=None, tups=None,
           tokenizer=None, model=None):
    # Use per-image dataset questions for human and machine.
    model.eval()
    test_q = [q for i, q in tups]

    #Very important that the boxes are normalized
    normalized_boxes = output_dict.get("normalized_boxes")
    x_attentions = dict()
    vis_attentions = dict()
    text_attentions = dict()
    features = output_dict.get("roi_features")
    answers = {v: "" for v in test_q}
    for ix, test_question in enumerate(test_q):
        inputs = tokenizer(
            test_question,
            padding="max_length",
            max_length=30,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        # run lxmert(s)
        output_vqa = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=True,
        )

        # get prediction
        pred_vqa = output_vqa["question_answering_score"].argmax(-1)
        vis_att = output_vqa["vision_attentions"]
        text_att = output_vqa["language_attentions"]
        x_att = output_vqa["cross_encoder_attentions"]
        print("Question:", test_question)
        x_attentions[test_question] = x_att
        vis_attentions[test_question] = vis_att
        text_attentions[test_question] = text_att
        answers[test_question] = vqa_answers[pred_vqa]
        print("Answer:", vqa_answers[pred_vqa])

    return x_attentions, vis_attentions, text_attentions, test_q

def attention_bbox_interpolation(im, bboxes, att, modality=None, hm=None, saveb=False):
    # https://github.com/facebookresearch/mmf/issues/145
    img_h, img_w = im.shape[:2]
    heatmap = np.zeros((img_h, img_w), np.float32)
    hmboxes = []
    if modality == "Cross":
        weights = att[0]
        for ix, (bbox, weight) in enumerate(zip(bboxes, weights)): # for each visual object weight
            x1, y1, x2, y2 = bbox
            hweights = hm[int(y1):int(y2), int(x1):int(x2)] # human attentions for detected object
            heatmap[int(y1):int(y2), int(x1):int(x2)] += weight
            hmboxes.append(hweights)
            if hweights.shape[0] == 0:
                print(hweights)

        hmboxes = np.array([hweights.sum()/hm.sum()
                            for hweights in hmboxes])
        # Cold temperature to get hardened, peakier distribution.
        hmboxes = softmax(hmboxes/0.1)
        if saveb:
            with open('{}hmboxes.npy'.format(DATA_PTH), 'wb') as f:
                np.save(f, hmboxes)
            with open('{}mboxes.npy'.format(DATA_PTH), 'wb') as f:
                np.save(f, weights)
        for hweights, weight, bbox in zip(hmboxes, weights, bboxes):
            x1, y1, x2, y2 = bbox
    elif modality == "Vision":
        for ix, (bbox, weights) in enumerate(zip(bboxes, att)):
            x1, y1, x2, y2 = bbox
            weight = weights[ix] # simply choose corresponding box attention
            if weight > 0.15:
                heatmap[int(y1):int(y2), int(x1):int(x2)] += weight
    cmap = matplotlib.cm.get_cmap('jet')
    cmap.set_bad(color="k", alpha=0.0)

    heatmap = np.minimum(heatmap, 1)
    heatmap = heatmap[..., np.newaxis]
    heatmap = np.array(imager.fromarray(cmap(heatmap, bytes=True), 'RGBA'))
    heatmap = heatmap.astype(im.dtype)
    result = cv2.addWeighted(im, 0.7, heatmap, 0.4, 0)
    result = result.astype(im.dtype)
    return result, hmboxes

def human_attention_interpolation(ip, att, mod):
    img = cv2.imread(ip)
    #  normalize to make the values in between 0 and 1
    heatmap = np.maximum(att, 0)
    heatmap /= np.max(heatmap)
    hmrsz = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    hmrsz2 = cv2.resize(att, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * hmrsz)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
    result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return result, hmrsz2

def attention_interpolation_from_saved(img, att):
    heatmap = np.uint8(255 * att)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
    result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return result

def visualize_pred(im_path, boxes, ref_vis_weights, ref_x_attentions,
                   ref_text_attentions, text_attentions,
                   x_attentions, vis_weights, model_type,
                   layer, head, modality='Cross',
                   average=False, h_att=None,
                   question="", tups=[], saveb=False,
                   view="head", tokenizer=None):

    if model_type == "Reference":
        x_attentions = ref_x_attentions
        vis_weights = ref_vis_weights
        text_attentions = ref_text_attentions
    if modality == 'Vision':
        att_weights = vis_weights[question]
    elif modality == "Cross":
        att_weights = x_attentions[question]
    elif modality== "Text":
            att_weights = text_attentions[question]

    if "http" in im_path:
        resp = urllib.request.urlopen(im_path)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        im = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        resp = imager.open(im_path)
        image = np.array(resp)
        # Convert RGB to BGR
        im = image[:, :, ::-1].copy()

    im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    b,g,r,a = cv2.split(im) # get b, g, r
    im = cv2.merge([r,g,b,a])
    if modality != "Text":
        d = {q: i for (i, q) in tups}
        h_att = np.reshape(h_att[d[question]], (14, 14))
        hum_ocr_att, hmrsz = human_attention_interpolation(im_path,
                                                           h_att,
                                                           mod=modality)
    if modality == "Human":
        fig, ax = plt.subplots()
        ax.imshow(hum_ocr_att)
        plt.show()
    elif modality == "Cross" or modality != "Text":
        if average:
            att_weights = torch.mean(att_weights[layer-1][0],
                                     dim=0).detach().numpy()
        else:
            att_weights = att_weights[layer-1][0][head-1].detach().numpy()
        if modality == "Cross":
            M = min(len(boxes), len(np.transpose(att_weights)))
            mod = "Cross"
        elif modality != "Text":
            M = min(len(boxes), len(att_weights))
            mod = "Vision"
        im_ocr_att, hmboxes = attention_bbox_interpolation(im=im,
                                                           bboxes=boxes[:M],
                                                           att=att_weights[:M],
                                                           modality=mod,
                                                           hm=hmrsz,
                                                           saveb=saveb)
        plt.imshow(im_ocr_att)
    else:
        tokens = tokenizer.encode(question,
                                  add_special_tokens=True,
                                  padding='max_length',
                                  max_length=30)
        tokens = tokenizer.convert_ids_to_tokens(tokens,
                                                 skip_special_tokens=True)
        attention = [att_weights[layer-1][:, :, :len(tokens)]]
        print()
        bertv(attention, tokens)

def call_html():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))

def bertv(attention, tokens):
    call_html()
    head_view(attention, tokens)
