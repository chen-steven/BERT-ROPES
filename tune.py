import os


def tune_baseline():
    os.system("mkdir train/new_yn_hotpot10")
    os.system("python main_hp.py --batch-size 2 --epochs 3 --learning-rate 3e-5 --seed 10 "
              "--output_dir train/new_yn_hotpot10 --gradient-accumulation-steps 4 ")


def tune_baseline_bce():
    os.system("mkdir train/hotpot10_roberta_bce")
    os.system("python main_hp.py --batch-size 2 --epochs 3 --learning-rate 3e-5 --seed 10 "
              "--output_dir train/hotpot10_roberta_bce --gradient-accumulation-steps 4 --binary ")


if __name__ == '__main__':
    tune_baseline()