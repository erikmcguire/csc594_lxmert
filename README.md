# CSC 594: Human-like Visual Question Answering with Multimodal Transformers

A class project for DePaul University, Fall 2020. Intended, as usual, for learning purposes: in this case to explore multimodality, multi-head self-attention, human data, et cetera. Please forgive any mistakes and misconceptions.

- [Training notebook](https://colab.research.google.com/drive/1jvcd8S5JgNPPBgXwQi6ti2txMxa5LpVP)
- [Evaluation notebook](https://colab.research.google.com/drive/1zsxaGkp-EetLgp1dRk6IdMv4oc7JmMRL)
- [Paper](https://github.com/erikmcguire/csc594_lxmert/blob/main/csc594-mcguire_erik-report.pdf)

## Abstract

Recently, research has been focusing on multimodal models which fuse image and language data to ground the learning of representations. One popular multimodal task is Visual Question Answering (VQA), which requires choosing the correct answer given an image and a question. In addition, datasets such as VQA-HAT (Human ATtention) enable researchers to study where human subjects attend to images when completing the VQA task. These data can also be used to supervise attention, inducing human biases in how machines attend to the same image-question pairs for the VQA task. In this work, we investigate the attention supervision of a multimodal transformer model, LXMERT, specifically its cross-modal attentions. We study the performance of the supervised model and compare the human and machine attentions. We find that performance is maintained despite successfully influencing the model to attend in a more human-like manner.
