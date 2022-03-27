# PARSE 
This is implementation of [Parse: Pairwise Alignment of Representations in Semi-Supervised EEG Learning for Emotion Recognition](https://arxiv.org/abs/2202.05400) in PyTorch (Version 1.11.0).

This repository contains the source code of our paper, using following datasets:



- [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html): 15 subjects participated experiments with videos as emotion stimuli (three emotions: positive/negative/neutral) and EEG was recorded with 62 channels at sampling rate of 1000Hz.

- [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html): 15 subjects participated experiments with videos as emotion stimuli (four emotions: happy/sad/neutral/fear).  62 EEG recordings were collected with a sampling frequency of 1000Hz.

- [SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html): 16 participants were involved in experiments with videos as emotion stimuli (five emotions: happy/sad/disgust/neutral/fear). 62 EEG recordings were collected with a sampling frequency of 1000Hz.


- [AMIGOS](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/readme.html): 37 participants completed experiments with both 16 short and 4 long video clips as stimuli. Continuous valence and arousal scores in the range of [-1, 1] were assigned by annotators to each of 20-second video snippets. The continuous scores were then transformed into binary classes (positive and negative) with a threshold of 0. A total of 14 EEG channels were recorded at a sampling rate of 128Hz.

## Prerequisites
Please follow the steps below in order to be able to train our models:


1 - Install Requirements

```
pip3 install -r ./requirements.txt
```

2 - Download dataset, then load data, proprocessing data, and perfom feature extraction. In SEED-series datasets, you can use the features (e.g. differential entropy) that have been released by the dataset provider. Neither the preprocessing and feature extraction scripts nor the extracted features have been shared by the AMIGOS dataset provider. Therefore, we provide the [data processing and feature extraction](./library/data_processing.py) code while strictly following the official dataset description and the original publication in which the dataset was published.


3 - Save the preprocessed data and EEG into separate folders (e.g., '/train/de/' and '/train/psd/'). Move EEG features and corresponding labels to the address shown in [here](./main.py#L279-L302). 

4 -  Usage:
Train PARSE by 10 labeled data of SEED-V dataset:  
```
CUDA_VISIBLE_DEVICES=0 python3 ./PARSE/main.py --manualSeed 0 --dataset SEED-V --method PARSE --n-labeled 10 --batch-size 8 --alpha 0.25
```
Train MixMatch by 25 labeled data of AMIGOS dataset:  
```
CUDA_VISIBLE_DEVICES=0 python3 ./PARSE/main.py --manualSeed 0 --dataset AMIGOS --method AdaMatch --n-labeled 25 --batch-size 64 --alpha 0.75 --threshold 0.6
```


 ## Document Description
 
- `\library\model`: model architecture 
- `\library\optmization`:  unsupervised loss weight, weight optmization, learning rate decay, etc. 
- `\library\train_loop`:  training step for our proposed [PARSE](./library/train_loop.py#L279-L391) and other three holistic semi-supervised methods ([MixMatch](./library/train_loop.py#L26-L98), [FixMatch](./library/train_loop.py#L105-L140) and [AdaMatch](./library/train_loop.py#L148-L271)) for EEG representation learning.
- `\main`: implementation of experiment set-up for several recent SSL pipelines and our method PARSE for all the four datasets. 
 


If you find this material useful, please cite the following article:

## Citation
```
@article{zhang2022parse,
  title={PARSE: Pairwise Alignment of Representations in Semi-Supervised EEG Learning for Emotion Recognition},
  author={Zhang, Guangyi and Etemad, Ali},
  journal={arXiv preprint arXiv:2202.05400},
  year={2022}
}
```
<img src="/architecture.jpg" width="900" height="350">



## Contact
Should you have any questions, please feel free to contact me at [guangyi.zhang@queensu.ca](mailto:guangyi.zhang@queensu.ca).




