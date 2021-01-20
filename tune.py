import os


def tune_baseline():
    os.system("mkdir train/hotpot10_roberta")
    os.system("python -u main_hp.py --batch-size 2 --epochs 3 --learning-rate 3e-5 --seed 10 "
              "--output_dir train/hotpot10_roberta --gradient-accumulation-steps 4 ")


if __name__ == '__main__':
    tune_baseline()