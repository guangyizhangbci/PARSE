# PARSE (PyTorch Version 1.11.0)
This is implementation of [Parse: pairwise alignment of representations in semi-supervised eeg learning for emotion recognition](https://arxiv.org/abs/2202.05400)

This repository contains the source code of our paper, using following datasets:

- Emotion Recoginition: 

    - [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html): 15 subjects participated experiments with videos as emotion stimuli (positive/negative/neutral) and EEG was recorded with 62 channels at sampling rate of 1000Hz.

    - [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html): 15 subjects participated experiments with videos as emotion stimuli (happy/sad/neutral/fear) and EEG was recorded with 62 channels at sampling rate of 1000Hz.

- Motor Imagery: 

    - [SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html): 16 participants were involved in experiments with videos as emotion stimuli (happy/sad/disgust/neutral/fear). 62 EEG recordings were collected with a sampling frequency of 1000Hz.


    - [AMIGOS](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/readme.html):37 participants completed experiments with both 16 short and 4 long video clips as stimuli. Valence and arousal scores are assigned by annotators to each of 20-second video snippets. 14 EEG channels were recorded at a sampling rate of
128Hz.

## Prerequisites
Please follow the steps below in order to be able to train our models:


1 - Install Requirements

```
pip3 install -r ./requirements.txt
```

2 - Download dataset, then load data, proprocessing data, and perfom feature extraction. Here is an [example](./library/data_processing.py).

3 - Save the preprocessed data and EEG into separate folders (e.g., '/train/de/' and '/train/psd/'). Move EEG features and corresponding labels to the address shown in [main](./main.py). 




 ## Document Description
 
- `\library\model`: 
- `\library\optmization`:  
- `\library\train_loop`:  
- `\main`: core code for implmentation of our proposed PARSE and other three holistic semi-supervised methods (MixMatch, FixMatch and AdaMatch) for EEG representation learning.     
 


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
<img src="/doc/architecture.jpg" width="1100" height="350">



## Contact
Should you have any questions, please feel free to contact me at [guangyi.zhang@queensu.ca](mailto:guangyi.zhang@queensu.ca).




