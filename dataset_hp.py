from torch.utils.data import Dataset
import json
import numpy as np
from tqdm import tqdm
import torch
from utils import HOTPOTExample
from evaluate_hp import normalize_answer
from transformers import BertTokenizerFast, AutoTokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')


HOTPOT_DATA_PATH = 'data/hotpot/'
MAX_PARAGRAPH_LEN = 400


class HOTPOT(Dataset):
    def __init__(self, tokenizer, file_path, eval=False, multi_label=False):
        self.tokenizer = tokenizer
        self.eval = eval
        examples, questions, contexts = get_examples(file_path)
        self.examples = examples
        self.encodings = convert_examples_to_features(examples, tokenizer, questions,
                                                      contexts, multi_label=multi_label)

    def __getitem__(self, idx):
        inputs = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.eval:
            inputs['id'] = self.examples[idx].qas_id
        return inputs

    def __len__(self):
        return len(self.examples)


def convert_examples_to_features(examples, tokenizer, questions, contexts, max_seq_length=512,
                                 doc_stride=1, multi_label=False):
    # TODO: also return features with ROPESFeatures object
    # features = []
    encodings = tokenizer(questions, contexts, padding=True, truncation=True)
    start_positions, end_positions = [], []
    start_labels, end_labels = [], []
    count, count1 = 0, 0
    for i, example in tqdm(enumerate(examples)):
        answer = example.answer
        question = example.question
        context = example.context
        q_idx = question.find(answer)
        c_idx = context.find(answer)
        question_tokens = tokenizer.tokenize(question)
        context_encoding = tokenizer(context)
        start_label, end_label = [0] * 512, [0] * 512
        start_ends = []
        if q_idx != -1:
            start_position = encodings.char_to_token(i, q_idx)
            end_position = encodings.char_to_token(i, q_idx + len(answer) - 1)
            if start_position < 512 and end_position < 512:
                start_label[start_position] = 1
                end_label[end_position] = 1
                if multi_label:
                    s_idx = 0
                    while True:
                        c_idx = context[s_idx:].find(answer)
                        if c_idx == -1:
                            break
                        c_idx += s_idx
                        sp = context_encoding.char_to_token(c_idx) + len(question_tokens) + 1
                        ep = context_encoding.char_to_token(c_idx + len(answer) - 1) + len(question_tokens) + 1
                        if sp >= 512 or ep >= 512:
                            break
                        start_label[sp] = 1
                        end_label[ep] = 1
                        start_ends.append((sp, ep))
                        s_idx = c_idx + len(answer)
        elif c_idx != -1:
            start_position = context_encoding.char_to_token(c_idx) + len(question_tokens) + 1
            end_position = context_encoding.char_to_token(c_idx + len(answer) - 1) + len(question_tokens) + 1
            if start_position < 512 and end_position < 512:
                start_label[start_position] = 1
                end_label[end_position] = 1
                if multi_label:
                    s_idx = c_idx + len(answer)
                    while True:
                        c_idx = context[s_idx:].find(answer)
                        if c_idx == -1:
                            break
                        c_idx += s_idx
                        sp = context_encoding.char_to_token(c_idx) + len(question_tokens) + 1
                        ep = context_encoding.char_to_token(c_idx + len(answer) - 1) + len(question_tokens) + 1
                        if sp >= 512 or ep >= 512:
                            break
                        start_label[sp] = 1
                        end_label[ep] = 1
                        start_ends.append((sp, ep))
                        s_idx = c_idx + len(answer)
        else:
            start_position = 0
            end_position = 0
        tmp = tokenizer.decode(encodings['input_ids'][i][start_position:end_position + 1])
        if normalize_answer(tmp) != normalize_answer(answer) and start_position < 512 and end_position < 512:
            print(tmp, answer)
            count1 += 1
        if start_position >= 512:
            start_position = 0
        if end_position >= 512:
            end_position = 0

        if start_position == end_position == 0:
            count += 1
        start_labels.append(start_label)
        end_labels.append(end_label)
        start_positions.append(start_position)
        end_positions.append(end_position)
    print(count, count1)
    encodings['start_positions'] = start_positions
    encodings['end_positions'] = end_positions
    encodings['start_labels'] = start_labels
    encodings['end_labels'] = end_labels
    return encodings


def get_examples(file_path):
    examples, questions, contexts = [], [], []
    count = 0
    with open(f'{HOTPOT_DATA_PATH}{file_path}', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for article in tqdm(data):
            id = article["_id"]
            question = article['question']
            answer = article["answer"]
            # remove yes or no questions
            if answer in ["yes", "no"]:
                continue
            supporting_facts = [title for title, _ in article['supporting_facts']]
            sup_paras, other_paras = [], []
            for i, (title, sents) in enumerate(article['context']):
                if title in supporting_facts:
                    sup_paras.append((i, ' '.join(sents)))
                else:
                    other_paras.append((i, ' '.join(sents)))
            for para in other_paras:
                if len(tokenizer.tokenize(' '.join([p for _, p in sup_paras] + [para[1]]))) > 512:
                    break
                sup_paras.append(para)
            sup_paras = sorted(sup_paras, key=lambda x: x[0])
            context = ' '.join([p for _, p in sup_paras])
            sit_idx = context.find(answer)
            q_idx = question.find(answer)
            if q_idx != -1:
                start_position = q_idx
            else:
                start_position = sit_idx
            if start_position == -1:
                count += 1
            example = HOTPOTExample(id, question, context, answer.strip(), start_position)
            examples.append(example)
            questions.append(question)
            contexts.append(context)
    print(len(examples), count)
    return examples, questions, contexts


if __name__ == '__main__':

    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    HOTPOT(tokenizer, 'hotpot_dev_distractor_v1.json', multi_label=True)
    # print('converting to features...')
    # convert_examples_to_features(examples, tokenizer, questions, contexts )

    # dataset = ROPES(tokenizer, 'dev-v1.0.json')
    # print(len(dataset[0]['input_ids']))
