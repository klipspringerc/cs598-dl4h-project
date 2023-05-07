# Reproducibility Project: Readmission Prediction via Deep Contextual Embedding of Clinical Concepts


Author: Cui Shengping, NetID: scui10, CS598 DL4H Spring 2023.

Code implementation for UIUC CS598 DL4H project.

## Env setup

- This project inherits the default python environment from CS598, so a ready CS598 env should be able to directly run all scripts in this repo.
- To set up from scratch:

```bash
conda create --name dlh python=3.8
conda activate dlh

# could use stable version instead. nightly build contain mps acceleration support
conda install pytorch torchvision torchaudio -c pytorch-nightly

pip install -r requirements.txt
```

# Notebook

Because many scripts involve loading and processing that could take significant amount of time,
it is recommended to perform reproducing steps in an interactive manner based on the [notebook](reproduce_content.ipynb).

The notebook describes step-by-step the data processing, implementation. training and evaluation of CONTENT model.

Swap the model part to reproduce result of other models.

* note that some models have unique collate function and training & evaluation functions in their respective scripts.

## Scripts Directory

- [content.py](content.py): CONTENT model implementation
- [retain.py](retain.py): RETAIN model implementation
- [attn.py](attn.py): Simple attention model implementation
- [rnn.py](rnn.py): Embedding + GRU model implementation
- [data.py](data.py): Data preprocessing
- [common.py](common.py): Common training functions. Some model would use unique functions in their respective scripts.
- [util.py](util.py): Data load & store
