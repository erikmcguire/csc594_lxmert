from dataclasses import field, dataclass
from typing import Optional
from tqdm import tqdm

from vqa_lxmert import VQA
from vqa_param import args
from vqa_paths import MODEL_PTH
from vqa_hat import *

import torch
import json
import os

# For logging predictions, labels in Trainer compute_loss() override.
from transformers.integrations import is_wandb_available

from transformers import (
    LxmertTokenizer,
    LxmertForQuestionAnswering,
    TrainingArguments,
    Trainer
  )

# Trainer compute_metrics() override for evaluation, prediction.
from sklearn.metrics import (
  accuracy_score,
  precision_recall_fscore_support as score
  )

# Required for Trainer prediction_step() override.
from transformers.trainer_pt_utils import (
    nested_concat,
    nested_detach
)

# Instantiate tokenizer.
pretrained = f"{MODEL_PTH}{args.load}"
tokenizer = LxmertTokenizer.from_pretrained(f"{MODEL_PTH}lxmert-base-uncased")

# Instantiate VQA object: creates datasets, provides HAT methods.
vqa = VQA(tokenizer)

@dataclass
class VQATrainingArguments(TrainingArguments):
    """Required subclass to add joint loss
       coefficient for hyperparameter search."""
    x_lmbda: Optional[float] = field(default=args.x_lmbda,
                                     metadata={"help":
                                               "VQA-HLAT loss trade-off."})

class VQATrainer(Trainer):
    def prediction_step(self, model, inputs,
                        prediction_loss_only,
                        ignore_keys = None):
        """Modified to use LXMERT QA score for logits."""
        has_labels = all(inputs.get(k) is not None
                         for k in self.label_names)
        inputs = {k: v for (k, v) in inputs.items()
                       if k != "question_id"}
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.args.fp16 and _use_native_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                loss = outputs[0].mean().detach()
                logits = outputs[1:2]
                # Limit slice to question_answering_score.
            else:
                loss = None
                # Limit slice to question_answering_score.
                logits = outputs[0:1]
            if self.args.past_index >= 0:
                self._past = (outputs[self.args.past_index
                              if has_labels
                              else self.args.past_index - 1])
                logits = logits[: self.args.past_index - 1] + logits[self.args.past_index :]
        if prediction_loss_only:
            return (loss, None, None)
        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name)
                                         for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None
        return (loss, logits, labels)

    def compute_loss(self, model, inputs):
        """
            Subclass which computes joint attention loss
            with human attention supervision from VQA-HLAT.
            Logs training predictions and labels to wandb if available.
        """
        device = vqa.device

        # Exclude attributes which model's .forward() can't use.
        minputs = {k: v for (k, v) in inputs.items()
                  if k not in ["h_att", "question_id"]}

        outputs = model(**minputs,
                        output_attentions=True,
                        return_dict=True)

        """
        # Use wandb.ai to log training labels, preds.
        if args.do_eval and is_wandb_available():
            import wandb
            # Encode questions, obtain top logits.
            sentences = [tokenizer.decode(ids.tolist(),
                                          skip_special_tokens=True)
                         for ids in inputs["input_ids"]]
            # Get lists of labels, top logits from input, output dicts.
            labels = inputs['labels']
            logits = outputs.question_answering_score
            predictions = [pred.item() for pred in logits.argmax(-1)]
            # Log huggingface run's predictions, ground truth at W&B.
            wandb.log({'predictions': dict(zip(sentences,predictions)),
                       'labels': dict(zip(sentences, labels))})
        """

        # Set main loss.
        loss = outputs.loss
        if args.x_lmbda != 0:
            # Compute human attention loss.
            x_atts = outputs.cross_encoder_attentions
            h_atts = inputs["h_att"].to(device)
            att_loss = args.x_lmbda * vqa.get_hat_loss(x_atts, h_atts)
            # Joint loss
            loss += att_loss
        return loss

# Run trainer evaluation on test set (w/ compute_metrics
# subclass) if desired:
eval_set = (vqa.test_dataset
            if not args.do_train and args.do_predict
            else vqa.val_dataset)

def evaluate(trainer, training_args):
    """Run evaluations and save attentions for comparison."""
    model_class = LxmertForQuestionAnswering
    model = model_class.from_pretrained(f"{MODEL_PTH}{args.load}")
    model.to(vqa.device)
    model.eval()

    for setname, dataset in [("eval", vqa.val_dataset),
                             ("test", vqa.test_dataset)]:
        # Reset model, human attentions for each set.
        vqa.all_att = {"model": [], "hat": []}

        result = [] # list of result dictionaries to save to json

        loader = trainer.get_eval_dataloader(dataset)
        for _, inputs in enumerate(tqdm(loader, desc="Evaluating")):

            # Exclude attributes which model's .forward() can't process.
            minputs = {k: v for (k, v) in inputs.items()
                      if k not in ["h_att", "question_id"]}
            minputs = trainer._prepare_inputs(minputs) # Move to GPU.

            with torch.no_grad():
                outputs = model(**minputs,
                                output_attentions=True,
                                return_dict=True)

                # Log batch's model, human attentions for export.
                x_atts = outputs.cross_encoder_attentions
                vqa.save_all_atts(x_atts, inputs["h_att"].to(vqa.device))

                # Process test set results for submission.
                if args.save_preds and args.do_predict:

                    # Get batch logits, convert samples' top preds to labels.
                    scores = outputs.question_answering_score
                    preds = scores.argmax(-1)
                    preds = np.array([eval_set.json_set.label2ans[l]
                                      for l in fl])

                    # Add batch's dictionaries to results for export.
                    result.extend([{'answer': pred,
                                    'question_id': qid.item()}
                                   for pred, qid in zip(preds,
                                                        inputs["question_id"])])

        # Save test set results for submission.
        with open(os.path.join(args.output_dir, 'result.json'), 'w') as jsn:
            json.dump(result, jsn, indent=4, sort_keys=True)

        # Save associated human, model attentions for set.
        torch.save(vqa.all_att,
                f'{training_args.output_dir}/all_atts_{setname}.pt')

def log_res(preds, labels):
    """
    Save dev, test predictions, labels to
    files, for significance testing.

    If args.do_eval only, saves dev set results.
    OR if args.do_predict and/or args.do_eval, saves test set.
    """
    ext = "_eval" if not args.do_predict else "_test"
    for n, fl in [("preds", preds), ("labels", labels)]:

        # Convert ids to answers.
        fl = np.array([eval_set.json_set.label2ans[l]
                       for l in fl])

        # Save associated validation predictions, labels.
        savedir = f"{args.output_dir}{ext}-{args.chkpt}"

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        np.savetxt(f"{savedir}/{n}.csv",
                   fl.astype(str),
                   fmt='%s',
                   delimiter=",")

def compute_metrics(pred):
    """
        Evaluation/prediction metrics subclass
        which may log predictions, labels to file
        for significance testing.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, support = score(labels, preds,
                                           average='macro',
                                           zero_division=0)
    acc = accuracy_score(labels, preds)

    # Save predictions, ground truth.
    log_res(preds,
            labels) if args.save_preds else None

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def model_init(trial):
    """
        Initializes models for hyperparameter searches.
    """
    model_class = LxmertForQuestionAnswering
    model = LxmertForQuestionAnswering.from_pretrained(pretrained)
    return model

def hp_space(trial):
    """
        Hyperparameter search over attention loss
        trade-off coefficients.
    """
    return {
        "x_lmbda":
        trial.suggest_float("x_lmbda",
                            0, 1.0,
                            step=0.2,
                            log=False),
    }
