import torch
import numpy as np
import random

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_ropes_examples():
    examples = []

class ROPESExample:
    def __init__(self, qas_id, question, context, answer, start_position):
        self.qas_id = qas_id
        self.question = question
        self.context = context
        self.answer = answer
        self.start_character = start_position
        self.end_position = start_position + len(answer)