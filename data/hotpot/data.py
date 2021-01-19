import numpy as np
import ujson as json
import nltk
nltk.download('punkt')
from nltk import word_tokenize

with open("hotpot_train_v1.1.json", 'r') as f:
    train = json.load(f)
with open("hotpot_dev_distractor_v1.json", 'r') as f:
    dev = json.load(f)

print(len(train))

lens = []
for item in train:
    all_sents = []
    context = item["context"]
    supporting_facts = [title for title, _ in item['supporting_facts']]
    for title, sents in context:
        if title not in supporting_facts:
            continue
        all_sents.extend(sents)
    print(len(word_tokenize(' '.join(all_sents))))
np.mean(lens)