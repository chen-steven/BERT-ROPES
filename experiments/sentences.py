import json
import torch
import math
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizerFast, BertModel

class SentenceSelection:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model = BertModel.from_pretrained('bert-base-cased', return_dict=True).to(self.device)
        self.sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def _get_sentence_representation(self, sentences):
        with torch.no_grad():
            inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: val.to(self.device) for key,val in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.pooler_output

    def get_similarity(self, sentences, questions):
        s_repr = self._get_sentence_representation(sentences)
        q_repr = self._get_sentence_representation(questions)
        if s_repr.size(0) != q_repr.size(0):
            q_repr = q_repr.expand(s_repr.size(0), -1)
        return self.sim_func(s_repr, q_repr)

    def get_k_most_similar(self, context, question, answer, k=None):
        sents = sent_tokenize(context)
        sim = self.get_similarity(sents, question+' '+answer)
        vals = [(val, i) for i,val in enumerate(sim)]
        vals.sort(reverse=True)

        if k:
            return sents, [i for _, i in vals[:k]]
        else:
            return sents, [i for _, i in vals]


def enforce_top_k_contains_answer(sents, idxs, answer, question, k):
    if question.find(answer) != -1:
        return idxs[:k]
    for i in idxs[:k]:
        if sents[i].find(answer) != -1:
            return idxs[:k]
        
    for i in idxs[k:]:
        if sents[i].find(answer) != -1:
            new_idxs = idxs[:k]
            new_idxs[-1] = i
            return new_idxs

    print(sents)
    print(answer)
    return idxs[:k]

def process(examples, path, k=5, enforce_contains_answer=True, p=None):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    s = SentenceSelection()
    processed_examples = []
    count = 0
    total = 0
    for example in tqdm(examples):
        answer = example.answer.strip()
        total += 1
        context = example.context
        sents, k_idxs = s.get_k_most_similar(context, example.question, example.answer.strip())

        if p:
            k = math.ceil(p*len(sents))
        if enforce_contains_answer:
            k_idxs = enforce_top_k_contains_answer(sents, k_idxs, answer, example.question, k)
        else:
            k_idxs = k_idxs[:k]

        new_example = dict(
            id=example.qas_id,
            sentences = sents,
            top_k = k_idxs,
            question=example.question,
            answer=example.answer
        )
        
        processed_examples.append(new_example)
   # print(count, total, len(processed_examples))
    with open(path, 'w') as f:
        json.dump(processed_examples, f)

if __name__ == '__main__':
    #s = SentenceSelection()
    # context = "Most chemical reactions within organisms would be impossible under the conditions in cells. For example, the body temperature of most organisms is too low for reactions to occur quickly enough to carry out life processes. Reactants may also be present in such low concentrations that it is unlikely they will meet and collide. Therefore, the rate of most biochemical reactions must be increased by a catalyst. A catalyst is a chemical that speeds up chemical reactions. In organisms, catalysts are called enzymes . Essentially, enzymes are biological catalysts. " + "Scientists have recently discovered two organisms deep in a cave in the hills of Paris. One organism, Boncho, has many enzymes. The other organism, Hojo, has very few enzymes. Currently, scientists are in the process of analyzing these new organisms to learn more about them."
    # question = "Which organism will have an easier time creating chemical reactions? Boncho"
    # sents = sent_tokenize(context)
    # print(len(sents))
    # for sent in sents:
    #     print(sent)
    import os
    print(os.getcwd())
    #print(s.get_similarity(sent_tokenize(context), question))
