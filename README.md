# Centaur: Robust Multimodal Fusion for Human Activity Recognition
[![This is an image](https://img.shields.io/badge/arXiv-2303.04636-darkred)](https://arxiv.org/abs/2303.04636)
[![This is an image](https://img.shields.io/badge/license-MIT-green)](https://github.com/Sanju-Xaviar/Centaur/blob/main/LICENSE.md)

This repository contains the implementation of the paper entitled "Centaur: Robust Multimodal Fusion for Human Activity Recognition".


## Directories
  * [Centaur](https://github.com/sustainable-computing/Centaur/tree/main/Centaur): Source code of Centaur's data cleaning module and self-attention CNN for human activity recognition
       * [ConvAttn](https://github.com/sustainable-computing/Centaur/tree/main/Centaur/ConvAttn): Self-attention CNN module
       * [DAE](https://github.com/sustainable-computing/Centaur/tree/main/Centaur/DAE): Data cleaning module
  * [Baselines](https://github.com/sustainable-computing/Centaur/tree/main/Baselines): Source code for Baselines of Data cleaning, Human activity recognition and End-to-end robust multimodal fusion
       * [Cleaning module](https://github.com/sustainable-computing/Centaur/tree/main/Baselines/Cleaning%20module): DAAE and VRAE for the cleaning module
       * [End-to-end Robust Multimodal Fusion](https://github.com/sustainable-computing/Centaur/tree/main/Baselines/End-to-end%20Robust%20Multimodal%20Fusion): SADeepSense and UniTS
       * [Human activity recognition module](https://github.com/sustainable-computing/Centaur/tree/main/Baselines/Human%20activity%20recognition%20module):DeepCNN and DeepConvLSTM
  * [Results](https://github.com/sustainable-computing/Centaur/tree/main/Results): Plotting of figures for the End-to-end Robust Multimodal fusion

## Instructions
 * To train Centaur's data cleaning module use [DE-train-PAMAP2.ipynb](https://github.com/sustainable-computing/Centaur/blob/main/Centaur/DAE/DE-train-PAMAP2.ipynb) for PAMAP2 dataset. Similarly Centaur can be trained on Opportunity and HHAR dataset by choosing the appropriate files from Centaur directory.
 * To train Centaur's self-attention CNN module for HAR use [Eval-PAMAP2-ConvAttn.ipynb](https://github.com/sustainable-computing/Centaur/blob/main/Centaur/ConvAttn/Eval-PAMAP2-ConvAttn.ipynb)
 * To analyze Centaur's End-to-end robust multimodal fusion performance on PAMAP2 dataset use [DE-test-PAMAP2.ipynb](https://github.com/sustainable-computing/Centaur/blob/main/Centaur/DAE/DE-test-PAMAP2.ipynb). You would need to insert the  paths generated after training the data cleaning and attention module for evaluation of the model.
 
## Datasets
Scripts to the preprocessed data can be found in the appropriate directories.
The original (not preprocessed) datasets can be found at the following links:

 * PAMAP2 https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
 * OPPORTUNITY https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition
 * HHAR https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition
 * mHealth https://archive.ics.uci.edu/dataset/319/mhealth+dataset
 * WISDM https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset


## Dependencies
Package       | Version
------------- | -------------
Python3       | 3.8.13
PyTorch       | 1.10.2
TensorFlow    | 2.8.0
scikit-learn  | 1.1.2

## License
Refer to the file [LICENCE](https://github.com/Sanju-Xaviar/Centaur/blob/main/LICENSE.md)

## Citation
Sanju Xaviar, Xin Yang and Omid Ardakanian. 2023. [Robust Multimodal Fusion for Human Activity Recognition](https://arxiv.org/abs/2303.04636), preprint.
```
@misc{https://doi.org/10.48550/arxiv.2303.04636,
  doi = {10.48550/ARXIV.2303.04636},
  url = {https://arxiv.org/abs/2303.04636}, 
  author = {Xaviar, Sanju and Yang, Xin and Ardakanian, Omid}, 
  keywords = {Machine Learning (cs.LG), Signal Processing (eess.SP), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS:  Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  title = {Robust Multimodal Fusion for Human Activity Recognition},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```


