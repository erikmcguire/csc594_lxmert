from tasks.vqa_trainer_data import (
    VQATorchDataset,
    VQADataset
    )

from vqa_param import args

from typing import Tuple, Union
from torch import Tensor
import torch.nn as nn
import torch
import copy

class VQA:
    """Instantiate dataset classes for splits.

       Provide methods to compute loss from
       human attention supervision and to
       save human attentions.
    """
    def __init__(self, tokenizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # To save human, model attentions.
        self.all_att = {"model": [], "hat": []}

        # To encode QA data.
        self.tokenizer = tokenizer

        # Instantiate datasets for splits.
        self.is_valid = False

        # May not load train set to save time for certain runs.
        self.train_dataset = (self.get_data(args.train)
                              if args.do_train else None)

        self.val_dataset = self.get_data(args.valid)

        if args.save_att or args.do_predict:
            # May use dev set to save time for certain runs.
            self.test_dataset = (self.val_dataset
                                 if args.valid == args.test
                                 else self.get_data(args.test))

    def load_data(self, splits: str):
        """Load JSON data (no image features)."""
        return VQADataset(splits)

    def get_data(self, arg: str):
        """Instantiate dataset w/ JSON, img features."""
        if 'train' not in arg and not args.save_att:
            self.is_valid = True
        else: # If training or saving attentions, use h_att.
            self.is_valid = False
        return VQATorchDataset(self.load_data(arg),
                               self.tokenizer,
                               self.is_valid)

    def prep_lang_att(self, x_att: Union[Tuple[Tensor], Tensor],
                      layer_handling: str = "last",
                      head_handling: str = "max") -> Tensor:
        """
        Cross encoder attentions are h x v, [h]: language, [v]: vision,
         so dims are # encoded question tokens by # of img objects/boxes.
        Select last layer (or avg) model cross encoder attentions tensor.
         Shape (batch_size x num_heads x seq_len [h] x seq_len [v]).
        Select [CLS] query row as attentions for all sequences, heads.
         Shape (batch_size x num_heads x 1 x seq_len [v]).
        Select each key tokens' [CLS]-based attentions from avg values
         over heads, or max value among heads, or last head values.
         Shape (batch_size x 1 x seq_len [v]).
        Return batch of [CLS] queries' weights  for tokens in sequence.
        """
        if layer_handling == "avg":
            x_att = torch.stack(x_att) # tuple to tensor
            x_att = torch.mean(x_att, 0) # avg'd layerwise
        elif layer_handling == "last":
            x_att = x_att[-1] # last layer
        # else every, so we called this with x_att: Tensor
        x_att = x_att[:, :, 0] # all sequences, all heads, cls weights
        if head_handling == 'avg':
            x_att = torch.mean(x_att, dim=1) # avg'd over heads
        elif head_handling == 'max':
            x_att = torch.max(x_att, dim=1).values # max over heads
        else:
            last_head = x_att.shape[1] - 1
            x_att = x_att[:, last_head, :]
        if not head_handling == "last": # renormalize
            x_att = x_att/x_att.sum()
        return x_att

    def get_hat_loss(self, x_atts, h_att):
        """
            Extract representative object attentions, compute
            KL Divergence with corresponding human attentions.
        """
        # For random supervision ablation, create random reference, normalize.
        random_h_att = torch.randn_like(h_att)
        h_att = (random_h_att/random_h_att.sum()
                 if not args.ablation == 'hat' else h_att)

        if not args.layer_handling == 'every':
            x_att = self.prep_lang_att(x_atts,
                                       layer_handling=args.layer_handling,
                                       head_handling=args.head_handling)
            assert(x_att.shape == h_att.shape)
            return nn.KLDivLoss(reduction='batchmean',
                                log_target=False)(x_att.log(),
                                                  h_att.float())
        else: # average loss per layer
            kldiv = torch.tensor(0.).to(self.device)
            for i in range(len(x_atts)):
                x_att = self.prep_lang_att(x_atts[i],
                                           layer_handling=args.layer_handling,
                                           head_handling=args.head_handling)
                assert(x_att.shape == h_att.shape)
                kldiv += nn.KLDivLoss(reduction='batchmean',
                                      log_target=False)(x_att.log(),
                                                        h_att.float())
            return kldiv/len(x_atts)

    def save_all_atts(self, x_atts, h_att):
        """
            Extract representative object attentions,
            append to dictionary to save each batch's
            attentions for similarity analysis.
        """
        x_att = self.prep_lang_att(x_atts,
                                   layer_handling=args.layer_handling,
                                   head_handling=args.head_handling)
        assert(x_att.shape == h_att.shape)
        self.all_att["model"].append(x_att)
        self.all_att["hat"].append(h_att)
