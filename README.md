# BERT ROPES Baseline
PyTorch implementation of a BERT ROPES baseline.
##Project Structure

```pytorch-template/
│
├── main.py - main script to start training and eval dev data after every epoch
│
├── dataset.py - torch Dataset and preprocessing for ROPES
│
├── evaluate.py - SQUAD evaluation script adapted for ROPES
│
├── utils.py - basic utilility functions
│
├── data/ - default directory for storing input data
```
##Fine-tune
To start training, run

`python -u main.py --gpu 0 --batch-size 8 --epochs 10 --learning-rate 1e-5 --seed 42`

