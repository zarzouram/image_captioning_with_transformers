# Image Captioning using Transformer

- [1. Introduction](#1-introduction)
- [2. Run](#2-run)
  - [2.1. **IMPORTANT NOTE**](#21-important-note)
  - [2.2. Requiremnts](#22-requiremnts)
  - [2.3. Create Dataset](#23-create-dataset)
  - [2.4. Train the model](#24-train-the-model)
  - [2.5. Testing the model](#25-testing-the-model)
  - [2.6. Analysis Notebook](#26-analysis-notebook)
- [3. The Model](#3-the-model)
  - [3.1. Introduction](#31-introduction)
  - [3.2. Framework](#32-framework)
  - [3.3. Training](#33-training)
  - [3.4. Inference](#34-inference)
- [4. References](#4-references)

## 1. Introduction

This repository hosts the course project for the "LT2326: Machine learning for
statistical NLP" Course. I used a transformer-based model to generate a caption
for images in this project. This task is known as the Image Captioning task.

The document will first show how to run the code; then, it will discuss the
model, its hyperparameters, loss, and performance metrics.  At the end of this
document, I will discuss the model performance.

This project is based on CPTR [[1]](#1) with some modifications as discussed
below. The project uses PyTorch as a deep learning framework.

## 2. Run

### 2.1. **IMPORTANT NOTE**

PyTorch 1.8 provide the tranformer attention avereged across the heads. My
impelemnetation needs the attention for each head, so I have changed the
PyTorch implementation. I changed `torch/nn/functional.py` line 4818 from

```python
return attn_output, attn_output_weights.sum(dim=1) num_heads
```

to

```python
return attn_output, attn_output_weights
```

### 2.2. Requiremnts

The code was tested using python 3.8.12. Use `pip install -r requirements.txt`
to install the required libraries.

### 2.3. Create Dataset

The dataset that I used is MS COCO 2017 [[2]](#2). The train images can be downloaded
from [here](http://images.cocodataset.org/zips/train2017.zip), validation
images from [here](http://images.cocodataset.org/zips/val2017.zip) and
the annotations from
[here](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).

`code/create_dataset.py` processes the images, tokenizes the captions text, and
creates the vocabulary dictionary. The code also randomly split the data into
train, validation, and test splits (We only have the train and validation
splits). Each of train, validation, and testing split contains 86300, 18494,
and 18493 images respectively.

In the dataset, each image has five or more captions. Five random captions are
selected during dataset preparation if more than five are available. Also, I
neglect the words which occur less than three times and replace them with the
"UNKOWN" special token. Other special tokens are used: the "start of the
sentence", the "end of the sentence", and the "pad" tokens.

The script saves the images in `hdf5` files, tokenized captions in
`JSON` files, and the vocabulary dictionary in a  `pth` file.

To run the code the following arguments are needed:

  1. `dataset_dir`: Parent directory contains the MS COCO files
  2. `json_train`: Relative path to the annotation file for the train split;
     relative to dataset_dir
  3. `json_val`: Relative path to the annotation file for the validation split;
     relative to dataset_dir
  4. `image_train`: Relative directory to the train images; relative to
     dataset_dir
  5. `image_val`: Relative directory to the validation images; relative to
     dataset_dir
  6. `output_dir`: Directory to save the output files
  7. `vector_dir`: Directory to the pre-trained embedding vectors files. The
     code expects that the directory contains the files for the pre-trained
     vectors supported by `torchtext.vocab`
  8. `vector_dim`: The used embedding dimensionality.
  9. `min_freq`: Minimum frequency needed to include a token in the vocabulary
  10. `max_len`: Minimum length for captions

You can run the code using the default values of the arguments above.

```
python code/create_dataset.py [ARGUMENT]
```

The code will save under the `output_dir` the following files:

  1. Three `hdf5` files cantain the images; one for each split:
     `train_images.hdf5`, `val_images.hdf5` and `test_images.hdf5`.
  2. Three `JSONS` files contain the tokenized captions after the encoding
     using the vocab dictionary: `train_captions.json`, `val_captions.json`,
     and `test_captions.json`.
  3. Three `JSONS` files contain the length for each caption:
     `train_lengthes.json`, `val_lengthes.json`, and `test_lengthes.json`.
  4. A `pth` for the created vocabulary dictionary: `vocab.pth`


### 2.4. Train the model

`code/run_train.py` expects the following arguments:

  1. `dataset_dir`: The parent directory contains the process dataset files. It
     is the same as the `output_dir` in [Section 2.3 Create
     Dataset](#23-create-dataset)
  2. `config_path`: Path for the configuration json file `onfig.json`
  3. `device`: either gpu or cpu
  4. `resume`: if train resuming is needed pass the checkpoint filename

Loss and evaluation metrics are tracked using Tensorboard. The path to tensoboard files is `logs/exp_0102.1513`.

You can run the code using the default values of the arguments above.

```
python code/run_train.py [ARGUMENT]
```

### 2.5. Testing the model

`code/inference_test.py` reads images from the test split and generats a description using beam search. The output
of this module is a pandas dataframe that holds the following:

  1. The generated caption
  2. Top-k generated captions
  3. Captions ground truth
  4. Transformer's Encoder-Decoder cross attention
  5. Evaluation metrics values: "bleu1, bleu2, bleu3, bleu4, gleu, meteor"

`code/inference_test.py` expects the following arguments:

  1. `dataset_dir`: The parent directory contains the process dataset files. It
     is the same as the `output_dir` in [Section 2.3 Create
     Dataset](#23-create-dataset)
  2. `save_dir`: Directory to save the output dataframe
  3. `config_path`: Path for the configuration json file `onfig.json`
  4. `checkpoint_name`: File name for the checkpoint model to be tested.

You can run the code using the default values of the arguments above.

```
python code/inference_test.py [ARGUMENT]
```

### 2.6. Analysis Notebook

`code/experiment.ipynb` holds some analysis I did on the model perfromance. Also, the visualization of
attention is done in the notebook. Both `GIF` and `PNG` images are generated and saved under `images/tests`.

Section 2.0 in the notbook presents randomly selected samples from the `images/tests` using `ipywidgets`. See an example below.

<img src="https://github.com/zarzouram/xformer_img_captnng/blob/main/images/report/firefox_H4tRhSVOok.gif" width="80%" padding="100px 100px 100px 100px">


## 3. The Model

### 3.1. Introduction

This project uses a transformer [[3]](#3) based model to generate a description
for images. This task is known as the Image Captioning task. Researchers used
many methodologies to approach this problem. One of these methodologies is the
encoder-decoder neural network [4]. The encoder transforms the source image
into a representation space; then, the decoder translates the information from
the encoded space into a natural language. The goal of the encoder-decoder is
to minimize the loss of generating a description from an image.

As shown in the survey done by MD Zakir Hossain et al. [[4]](#4), we can see that the
models that use encoder-decoder architecture mainly consist of a language model
based on LSTM [[5]](#5), which decodes the encoded image received from a CNN, see
Figure 1.  The limitation of LSTM with long sequences and the success of
transformers in machine translation and other NLP tasks attracts attention to
utilizing it in machine vision. Alexey Dosovitskiy et al. introduce an image
classification model (ViT) based on a classical transformer encoder showing a
good performance [[6]](#6). Based on ViT, Wei Liu et al. present an image captioning
model (CPTR) using an encoder-decoder transformer [[1]](#1). The source image is fed
to the transformer encoder in sequence patches. Hence, one can treat the image
captioning problem as a machine translation task.

<img
src="https://github.com/zarzouram/xformer_img_captnng/blob/main/images/report/Encoder-Decoder.png"
width="80%" padding="100px 100px 100px 10px">

Figure 1: Encoder Decoder Architecture

### 3.2. Framework

The CPTR [[1]](#1) consists of an image patcher that converts images
![x\in\mathbb{R}^{H\times W\times
C}](https://latex.codecogs.com/svg.latex?x\in\mathbb{R}^{H\times%20W\times%20C})
to a sequence of patches ![x_p\in\mathbb{R}^{N(P^2\times
E)}](https://latex.codecogs.com/svg.latex?x_p\in\mathbb{R}^{N(P^2\times%20E)}),
where _N_ is number of patches, _H_, _W_, _C_ are images height, width and
number of chanel _C=3_ respectively, _P_ is patch resolution, and _E_ is image
embeddings size. Position embeddings are then added to the images patches,
which form the input to twelve layers of identical transformer encoders. The
output of the last encoder layer goes to four layers of identical transformer
decoders. The decoder also takes words with sinusoid positional embedding.

The pre-trained ViT weights initialize the CPTR encoder [[1]](#1). I omitted
the initialization and image positional embeddings, adding an image embedding
module to the image patcher using the features map extracted from the Resnet101
network [[7]](#7). The number of encoder layers is reduced to two. For
Resenet101, I deleted the last two layers and the last softmax layer used for
image classification.

Another modification takes place at the encoder side. The feedforward network
consists of two convolution layers with a RELU activation function in between.
The encoder side deals solely with the image part, where it is beneficial to
exploit the relative position of the features we have. Refer to Figure 2 for
the model architecture.

<img
src="https://github.com/zarzouram/xformer_img_captnng/blob/main/images/report/Architectures.png"
width="80%" padding="100px 100px 100px 10px">

Figure 2: Model Architecture

### 3.3. Training

The transformer decoder output goes to one fully connected layer, which
provides –-given the previous token–- a probability distribution
(![\in\mathbb{R}^k](https://latex.codecogs.com/svg.latex?\in\mathbb{R}^k), *k*
is vocabulary size) for each token in the sequence.

I trained the model using cross-entropy loss given the target ground truth
(![y_{1:T}](https://latex.codecogs.com/svg.latex?y_{1:T})) where _T_ is the
length of the sequence. Also, I add the doubly stochastic attention
regularization [[8]](#8) to the cross-entropy loss to penalize high weights in
the encoder-decoder attention. This term encourages the summation of attention
weights across the sequence to be approximatively equal to one. By doing so,
the model will not concentrate on specific parts in the image when generating a
caption. Instead, it will look all over the image, leading to a richer and more
descriptive text [[8]](#8).

The loss function is defined as:

![\large L=-\sum_{c=1}^{T}{log\left(p\left(y_c\middle| y_{c-1}\right)\right)\ +\ \sum_{l=1}^{L}{\frac{1}{L}\left(\sum_{d=1}^{D}\sum_{i=1}^{P^2}\left(1-\sum_{c=1}^{T}\alpha_{cidl}\right)^2\right)}}](https://latex.codecogs.com/svg.latex?\large%20L=-\sum_{c=1}^{T}{log\left(p\left(y_c\middle|%20y_{c-1}\right)\right)\%20+\%20\sum_{l=1}^{L}{\frac{1}{L}\left(\sum_{d=1}^{D}\sum_{i=1}^{P^2}\left(1-\sum_{c=1}^{T}\alpha_{cidl}\right)^2\right)}})

where _D_ is the number of heads and _L_ is the number of layers.

I used Adam optimizer, with a batch size of thirty-two. The reader can find the
model sizes in the configuration file `code/config.json`. Evaluation metrics
used are Bleu [[9]](#9), METEOR [[10]](#10), and Gleu [[11]](#11).

I trained the model for one hundred epochs, with stopping criteria if the
tracked evaluation metric (bleu-4) does not improve for twenty successive
epochs. Also, the learning rate is reduced by 0.25% if the tracked evaluation
metric (bleu-4) does not improve for ten consecutive epochs. The evaluation of
the model against the validation split takes place every two epochs.

The pre-trained Glove embeddings [[12]](#12) initialize the word embedding
weights. The words embeddings are frozen for ten epochs. The Resnet101 network
is tuned from the beginning.

### 3.4. Inference

A beam search of size five is used to generate a caption for the images in the
test split. The generation starts by feeding the image and the "start of
sentence" special tokens. Then at each iteration, five tokens with the highest
scores are chosen. The generation iteration stops when the "end of sentence" is
generated or the max length limit is reached.


## 4. References

<a id="1">[1]</a> Liu, W., Chen, S., Guo, L., Zhu, X., & Liu, J. (2021). CPTR:
Full transformer network for image captioning. arXiv preprint
[arXiv:2101.10804](https://arxiv.org/abs/2101.10804).

<a id="2">[2]</a> Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P.,
Ramanan, D., ... & Zitnick, C. L. (2014, September). Microsoft coco: Common
objects in context. In European conference on computer vision (pp. 740-755).
Springer, Cham.

<a id="3">[3]</a> A. Vaswani et al., 'Attention is all you need', Advances in neural
information processing systems, vol. 30, 2017.

<a id="4">[4]</a> M. Z. Hossain, F. Sohel, M. F. Shiratuddin, and H. Laga, 'A Comprehensive
Survey of Deep Learning for Image Captioning', arXiv:1810.04020 [cs, stat],
Oct. 2018, Accessed: Mar. 03, 2022. [Online]. Available:
http://arxiv.org/abs/1810.04020.

<a id="5">[5]</a> S. Hochreiter and J. Schmidhuber, ‘Long short-term memory’, Neural
computation, vol. 9, no. 8, pp. 1735–1780, 1997.

<a id="6">[6]</a> A. Dosovitskiy et al., 'An image is worth 16x16 words: Transformers for
image recognition at scale', arXiv preprint arXiv:2010.11929, 2020.

<a id="7">[7]</a> K. He, X. Zhang, S. Ren, and J. Sun, 'Deep Residual Learning for Image
Recognition', arXiv:1512.03385 [cs], Oct. 2015, Accessed: Mar. 06, 2022.
[Online]. Available: http://arxiv.org/abs/1512.03385.

<a id="8">[8]</a> K. Xu et al., 'Show, Attend and Tell: Neural Image Caption Generation with
Visual Attention', arXiv:1502.03044 [cs], Apr. 2016, Accessed: Mar. 07, 2022.
[Online]. Available: http://arxiv.org/abs/1502.03044.

<a id="9">[9]</a> K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, 'Bleu: a method for
automatic evaluation of machine translation', in Proceedings of the 40th annual
meeting of the Association for Computational Linguistics, 2002, pp. 311–318.

<a id="10">[10]</a> S. Banerjee and A. Lavie, 'METEOR: An automatic metric for MT evaluation
with improved correlation with human judgments', in Proceedings of the acl
workshop on intrinsic and extrinsic evaluation measures for machine translation
and/or summarization, 2005, pp. 65–72.

<a id="11">[11]</a> A. Mutton, M. Dras, S. Wan, and R. Dale, 'GLEU: Automatic evaluation of
sentence-level fluency', in Proceedings of the 45th Annual Meeting of the
Association of Computational Linguistics, 2007, pp. 344–351.

<a id="12">[12]</a> J. Pennington, R. Socher, and C. D. Manning, 'Glove: Global vectors for
word representation', in Proceedings of the 2014 conference on empirical
methods in natural language processing (EMNLP), 2014, pp. 1532–1543.

<a id="13">[13]</a> A. Karpathy and L. Fei-Fei, 'Deep visual-semantic alignments for
generating image descriptions', in Proceedings of the IEEE conference on
computer vision and pattern recognition, 2015, pp. 3128–3137.
