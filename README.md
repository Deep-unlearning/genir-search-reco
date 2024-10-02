# Genir Search Reco

## Overview

We propose a new model unified framework that integrates Recommender Systems (RS) and Product Search using Generative Information Retrieval (GenIR).
The proposed approach utilizes models like T5 to generate item identifiers from user queries, simplifying the retrieval process and enhancing personalized recommendations. 

## Requirements

```
accelerate==0.31.0
datasets==2.20.0
numpy==2.1.1
pandas==2.2.3
Requests==2.32.3
scikit_learn==1.5.2
sentence_transformers==3.0.1
torch==2.3.1
tqdm==4.66.4
transformers==4.44.0
wandb==0.18.2
```

## Datasets and weights

We use the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) datasets. The dataset can be downloaded and is also available on https://hue.prod.crto.in/hue/filebrowser/view=%2Fuser%2Fs.zheng in the Video_Games folder. The weight of the RQ-VAE model and T5-small model are also available in the folder 4x256-0.0-0.0-0.0-0.0 and ckpt respectively.

- The dataset should be located in genir-search-reco/RQ-VAE/ID_generation/preprocessing/processed/
- The RQ-VAE weight should be located in genir-search-reco/RQ-VAE/results/{dataset}/
- The T5-small weught should be located in genir-search-reco/ 

## Train RQ-VAE

Go to RQ-VAE folder 

### Use Amazon 2023 Review
- Run process_amazon_2023.py to download [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) datasets.
- Run ./run_32d.sh to train the RQ-VAE
- Run generate_indices.py to generate the indices
- Run process_indices_amazon2023.ipynb notebook to process the indices
- Run process_search2023.ipynb notebook to process the search dataset.


## Train reco and serach

Go back to genir-search-reco

- run run_t5.sh to train the model

## Test the model

- run run_test.py on a single GPU or run_test_ddp.py on a multi-gpu setup
