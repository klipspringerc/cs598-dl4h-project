# Reproducibility Project: Readmission Prediction via Deep Contextual Embedding of Clinical Concepts

Author: Cui Shengping, NetID: scui10, CS598 DL4H Spring 2023.

Code implementation for UIUC CS598 DL4H project.

Reproduced paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0195024

## Env Setup

- This project inherits the default python environment from CS598, so a ready CS598 env should be able to directly run
  all scripts in this repo.
- Some dependencies include non-python libraries that are better installed via conda instead of pip.
- To set up from scratch:

```bash
conda create --name dlh python=3.8
conda activate dlh

# could use stable version instead. nightly build contain mps acceleration support
conda install pytorch torchvision torchaudio -c pytorch-nightly

pip install -r requirements.txt
```

# Data and Intermediate Results

- [S1_file](data/S1_File.txt): the raw EHR synthetic data.
- [s1_sorted.csv](resource/s1_sorted.csv): original CSV sorted by patient and datetime
- [X_*.pkl](resource/): preprocessed sequence data with numerical diagnosis id.
- [Y_*.pkl](resource/): preprocessed sequence labels.

# Notebook

All model scripts involve loading and processing that could take significant amount of time.
It is recommended to perform reproducing steps in an interactive manner based on the [notebook](reproduce_content.ipynb).

The notebook describes step-by-step the data processing, implementation. training and evaluation of CONTENT model.

Import the model-specific functions from respective model script to reproduce result of other models.

If the below functions are present in the model script, then they are unique for this model.
Use the version in model script instead of the common version:

* `collate_fn`: content & rnn use multi-hot encoding; simple attention model enforces maximum sequence length; retain
  does not use multi-hot encoding.
* `train`, `eval`, `full_eval`: content & rnn use normal ordering; simple attention model uses reverse order input only;
  retain uses both


| model     | substitute functions                                   | batch size |
|-----------|--------------------------------------------------------|------------|
| CONTENT   | -                                                      | 16         |
| RETAIN    | `load_seq`, `collate_fn`, `train`, `eval`, `full_eval` | 32         |
| GRU       | -                                                      | 16         |
| SimpAttn  | `collate_fn`, `train`, `eval`, `full_eval`             | 16         |


## Model Scripts Directory

- [content.py](content.py): CONTENT model implementation
- [retain.py](retain.py): RETAIN model implementation
- [attn.py](attn.py): Simple Attention model implementation
- [rnn.py](rnn.py): Embedding + GRU model implementation
- [data.py](data.py): Data preprocessing, load & store
- [common.py](common.py): Common data, training and evaluation functions. Some models would use unique functions in their respective scripts instead.
- [util.py](util.py): General utils

## Model Snapshots

In [models](models) directory persists the optimal version state dict for each model.
Evaluation based on these snapshots should produce results similar to the project report.

- [content_opt.pth](models/content_opt.pth)
- [retain_opt.pth](models/retain_opt.pth)
- [simple_attn_opt.pth](models/simple_attn_opt.pth)
- [rnn_opt.pth](models/rnn_opt.pth)

### Model Evaluation

| model    | roc-auc | pr-auc |
|----------|---------|--------|
| CONTENT  | 0.8089  | 0.6754 |
| RETAIN   | 0.7846  | 0.6302 |
| GRU      | 0.7895  | 0.6530 |
| SimpAttn | 0.7944  | 0.6478 |

To evaluate the content model using test data:

```python
import torch
from content import Content
from common import full_eval, get_test_mhc_loader, collate_fn

# use respective collate function and eval function according to model description above.
# get_test_loader is an example for loading content dataset
num_codes = 492
test_loader = get_test_mhc_loader(collate_fn)

# when evaluating models, first import and init respective model object, then load the state dict
model = Content(input_dim=num_codes - 1)
model.load_state_dict(torch.load("models/content_opt.pth"))
p, r, f, roc_auc, pr_auc = full_eval(model, test_loader)
print('Test p: {:.4f}, r:{:.4f}, f: {:.4f}, roc_auc: {:.4f}, pr_auc: {:.4f}'.format(p, r, f, roc_auc, pr_auc))
```

Example A: substitute model specific function `collate_fn` and `full_eval` for simple attention model evaluation.

```python
import torch
from attn import SimpleAttn, collate_fn, full_eval
from common import get_test_mhc_loader

num_codes = 492
test_loader = get_test_mhc_loader(collate_fn)

model = SimpleAttn(input_dim=num_codes - 1)
model.load_state_dict(torch.load("models/simple_attn_opt.pth"))
p, r, f, roc_auc, pr_auc = full_eval(model, test_loader)
print('Test p: {:.4f}, r:{:.4f}, f: {:.4f}, roc_auc: {:.4f}, pr_auc: {:.4f}'.format(p, r, f, roc_auc, pr_auc))
```