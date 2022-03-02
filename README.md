# Image Captioning using Transformer

- [1. Introduction](#1-introduction)
- [2. Run](#2-run)
  - [2.1. Important note](#21-important-note)
  - [2.2. Requiremnts](#22-requiremnts)
  - [2.3. Create Dataset](#23-create-dataset)
  - [2.4. Train the model](#24-train-the-model)
  - [2.5. Testing the model](#25-testing-the-model)
  - [Analysis Notebook](#analysis-notebook)
- [3. References](#3-references)

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

### 2.1. Important note

note

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
splits). Then, it saves the images in `hdf5` files, tokenized captions in
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

### Analysis Notebook

`code/experiment.ipynb` holds some analysis I did on the model perfromance. Also, the visualization of 
attention is done in the notebook. Both `GIF` and `PNG` images are generated and saved under `images/tests`.

Section 2.0 in the notbook presents randomly selected samples from the `images/tests` using `ipywidgets`. See an example below.

<img src="https://github.com/zarzouram/xformer_img_captnng/blob/main/images/report/firefox_H4tRhSVOok.gif" width="80%" padding="100px 100px 100px 100px">


## 3. References

<a id="1">[1]</a>
Liu, W., Chen, S., Guo, L., Zhu, X., & Liu, J. (2021). Cptr: Full transformer
network for image captioning. arXiv preprint
[arXiv:2101.10804](https://arxiv.org/abs/2101.10804).


<a id="2">[2]</a>
Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. (2014, September). Microsoft coco: Common objects in context. In European conference on computer vision (pp. 740-755). Springer, Cham.
