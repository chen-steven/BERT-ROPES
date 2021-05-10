from torch.utils.data import Dataset
import json
import numpy as np
from tqdm import tqdm
import torch
from utils import HOTPOTExample
from evaluate_hp import normalize_answer
from collections import Counter
from transformers import BertTokenizerFast, AutoTokenizer, RobertaTokenizerFast, LongformerTokenizerFast
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
# tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-large-4096', cache_dir="train/cache")

HOTPOT_DATA_PATH = 'data/hotpot/'
MAX_PARAGRAPH_LEN = 400


class HOTPOT(Dataset):
    def __init__(self, tokenizer, file_path, eval=False, multi_label=False, max_seq_length=512):
        self.tokenizer = tokenizer
        self.eval = eval
        examples, questions, contexts = get_examples(file_path)
        self.examples = examples
        self.encodings = convert_examples_to_features(examples, tokenizer, questions,
                                                      contexts, multi_label=multi_label, max_seq_length=max_seq_length)

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
        length = len(encodings['input_ids'][i])
        start_label, end_label = [0] * length, [0] * length
        start_ends = []
        if q_idx != -1:
            start_position = encodings.char_to_token(i, q_idx)
            end_position = encodings.char_to_token(i, q_idx + len(answer) - 1)
            if start_position < length and end_position < length:
                start_label[start_position] = 1
                end_label[end_position] = 1
                if multi_label:
                    s_idx = 0
                    while True:
                        c_idx = context[s_idx:].find(answer)
                        if c_idx == -1:
                            break
                        c_idx += s_idx
                        sp = context_encoding.char_to_token(c_idx) + len(question_tokens) + 2
                        ep = context_encoding.char_to_token(c_idx + len(answer) - 1) + len(question_tokens) + 2
                        if sp >= length or ep >= length:
                            break
                        start_label[sp] = 1
                        end_label[ep] = 1
                        start_ends.append((sp, ep))
                        s_idx = c_idx + len(answer)
        elif c_idx != -1:
            start_position = context_encoding.char_to_token(c_idx) + len(question_tokens) + 2
            end_position = context_encoding.char_to_token(c_idx + len(answer) - 1) + len(question_tokens) + 2
            if start_position < length and end_position < length:
                start_label[start_position] = 1
                end_label[end_position] = 1
                if multi_label:
                    s_idx = c_idx + len(answer)
                    while True:
                        c_idx = context[s_idx:].find(answer)
                        if c_idx == -1:
                            break
                        c_idx += s_idx
                        sp = context_encoding.char_to_token(c_idx) + len(question_tokens) + 2
                        ep = context_encoding.char_to_token(c_idx + len(answer) - 1) + len(question_tokens) + 2
                        if sp >= length or ep >= length:
                            break
                        start_label[sp] = 1
                        end_label[ep] = 1
                        start_ends.append((sp, ep))
                        s_idx = c_idx + len(answer)
        else:
            start_position = 0
            end_position = 0
        tmp = tokenizer.decode(encodings['input_ids'][i][start_position:end_position + 1], skip_special_tokens=True)
        if normalize_answer(tmp) != normalize_answer(answer) and start_position < length and end_position < length:
            # print(tmp, answer)
            count1 += 1
        if start_position >= length:
            start_position = 0
        if end_position >= length:
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
            id = article["id"]
            question = article['question']
            answer = article["answer"]
            context = []
            for title, sents in article['context']:
                context.extend(sents)
            context = ' '.join(context)
            sit_idx = context.find(answer)
            q_idx = question.find(answer)
            if q_idx != -1:
                start_position = q_idx
            else:
                start_position = sit_idx
            if start_position == -1:
                count += 1
            # print(id, question, context, answer, start_position)
            example = HOTPOTExample(id, question, context, answer.strip(), start_position)
            examples.append(example)
            questions.append(question)
            contexts.append(context)
    print(len(examples), count)
    return examples, questions, contexts


def preprocess_hotpot(file_path):
    new_data = []
    with open(f'{HOTPOT_DATA_PATH}{file_path}', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for article in tqdm(data):
            id = article["_id"]
            question = article['question']
            answer = article["answer"]
            # remove yes or no questions
            if answer in ["yes", "no"]:
                continue
            supporting_facts = set([title for title, _ in article['supporting_facts']])
            sup_paras, other_paras = [], []
            for i, (title, sents) in enumerate(article['context']):
                if title in supporting_facts:
                    sup_paras.append((i, title, sents))
                else:
                    other_paras.append((i, title, sents))
            np.random.seed(42)
            np.random.shuffle(other_paras)
            assert len(sup_paras) == 2
            paras = sup_paras
            # if len(other_paras) > 0:
            #     paras = sup_paras + [other_paras[0]] # supporting documents + one other document
            paras = sorted(paras, key=lambda x: x[0])

            new_data.append({
                "id": id,
                "question": question,
                "answer": answer,
                "supporting_facts": article['supporting_facts'],
                "context": [[para[1], para[2]] for para in paras]
            })

    with open(f'{HOTPOT_DATA_PATH}new1_{file_path}', 'w') as f:
        json.dump(new_data, f)


def preprocess_hotpot_new(file_path):
    new_data = []
    counts = []
    with open(f'{HOTPOT_DATA_PATH}{file_path}', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for article in tqdm(data):
            id = article["_id"]
            question = article['question']
            answer = article["answer"]
            # remove yes or no questions
            counts.append(len(article['context']))
            if answer in ["yes", "no"]:
                continue
            supporting_facts = set([title for title, _ in article['supporting_facts']])
            sup_paras, add_other_paras, other_paras = [], [], []
            existing_titles = set()
            for i, (title, sents) in enumerate(article['context']):
                if title in supporting_facts:
                    if title in existing_titles:
                        continue
                    existing_titles.add(title)
                    sup_paras.append((i, title, sents))
                else:
                    if "[added" in title:
                        add_other_paras.append((i, title, sents))
                    else:
                        other_paras.append((i, title, sents))
            np.random.seed(42)
            np.random.shuffle(add_other_paras)
            np.random.shuffle(other_paras)
            assert len(sup_paras) == 2
            paras = sup_paras
            if len(add_other_paras) > 0:
                paras = sup_paras + [add_other_paras[0]]  # supporting documents + one other document
            elif len(other_paras) > 0:
                paras = sup_paras + [other_paras[0]]  # supporting documents + one other document
            try:
                assert len(paras) == 3
            except:
                print(len(paras), len(add_other_paras), len(other_paras))
            paras = sorted(paras, key=lambda x: x[0])
            new_data.append({
                "id": id,
                "question": question,
                "answer": answer,
                "supporting_facts": article['supporting_facts'],
                "context": [[para[1], para[2]] for para in paras]
            })
    print(len(new_data))
    print(Counter(counts))
    with open(f'{HOTPOT_DATA_PATH}new_yn_{file_path}', 'w') as f:
        json.dump(new_data, f)


def preprocess_hotpot_rebuttal(file_path):
    new_data = []
    counts = []
    with open(f'{HOTPOT_DATA_PATH}{file_path}', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for article in tqdm(data):
            id = article["_id"]
            question = article['question']
            answer = article["answer"]

            if answer in ["yes", "no"]:
                question = f"yes no {question}"
                counts.append(len(article['context']))
                supporting_facts = set([title for title, _ in article['supporting_facts']])
                sup_paras, add_other_paras, other_paras = [], [], []
                existing_titles = set()
                for i, (title, sents) in enumerate(article['context']):
                    if title in supporting_facts:
                        if title in existing_titles:
                            continue
                        existing_titles.add(title)
                        sup_paras.append((i, title, sents))
                    else:
                        if "[added" in title:
                            add_other_paras.append((i, title, sents))
                        else:
                            other_paras.append((i, title, sents))
                np.random.seed(42)
                np.random.shuffle(add_other_paras)
                np.random.shuffle(other_paras)
                assert len(sup_paras) == 2
                paras = sup_paras
                if len(add_other_paras) > 0:
                    paras = sup_paras + [add_other_paras[0]]  # supporting documents + one other document
                elif len(other_paras) > 0:
                    paras = sup_paras + [other_paras[0]]  # supporting documents + one other document
                try:
                    assert len(paras) == 3
                except:
                    print(len(paras), len(add_other_paras), len(other_paras))
                paras = sorted(paras, key=lambda x: x[0])
                new_data.append({
                    "id": id,
                    "question": question,
                    "answer": answer,
                    "supporting_facts": article['supporting_facts'],
                    "context": [[para[1], para[2]] for para in paras]
                })
    print(len(new_data))
    print(Counter(counts))
    with open(f'{HOTPOT_DATA_PATH}new_{file_path}', 'r') as f:
        old_data = json.load(f)
    for example in old_data:
        example["question"] = f"yes no {example['question']}"
        new_data.append(example)
    print(len(new_data))
    # with open(f'{HOTPOT_DATA_PATH}new_yn_{file_path}', 'w') as f:
    #     json.dump(new_data, f)

def preprocess_hotpot_retrieved_docs(file_path, retrieved_docs_path):
    retrieved_paras = json.load(open(retrieved_docs_path, 'r'))
    for key in retrieved_paras:
        l = []
        for p in retrieved_paras[key]:
            l.extend(p)
        retrieved_paras[key] = l[:3]

    new_data = []
    counts = []
    with open(f'{HOTPOT_DATA_PATH}{file_path}', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for article in tqdm(data):
            id = article["_id"]
            question = article['question']
            answer = article["answer"]

            question = f"yes no {question}"
            counts.append(len(article['context']))
            supporting_facts = set([title for title in retrieved_docs_path[id]])
            sup_paras, add_other_paras, other_paras = [], [], []
            for i, (title, sents) in enumerate(article['context']):
                if title in supporting_facts:
                    sup_paras.append((i, title, sents))

            paras = sup_paras
            paras = sorted(paras, key=lambda x: x[0])
            new_data.append({
                "id": id,
                "question": question,
                "answer": answer,
                "supporting_facts": article['supporting_facts'],
                "context": [[para[1], para[2]] for para in paras]
            })
    print(len(new_data))

    with open(f'{HOTPOT_DATA_PATH}retrieved_yn_{file_path}', 'w') as f:
         json.dump(new_data, f)


def check_truncation(file_path):
    with open(f'{HOTPOT_DATA_PATH}{file_path}', 'r') as f:
        data = json.load(f)
    count = 0
    for example in tqdm(data):
        supporting_facts = set([title for title, _ in example['supporting_facts']])
        sents = []
        for _, ss in example["context"]:
            sents.extend(ss)

        words = tokenizer.tokenize(' '.join([example["question"]] + sents))
        if example["context"][-1][0] in supporting_facts and len(words) > 512:
            count += 1
    print(count)


if __name__ == '__main__':
    #check_truncation("new_hotpot_dev_distractor_v1.json")
    preprocess_hotpot_retrieved_docs('hotpot_train_v1.1.json', '')
    preprocess_hotpot_retrieved_docs('hotpot_dev_distractor_v1.json', '')
    # preprocess_hotpot_rebuttal('hotpot_train_v1.1.json')
    # preprocess_hotpot_rebuttal('hotpot_dev_distractor_v1.json')
    # preprocess_hotpot_rebuttal('hotpot_dev_distractor_v1_addDoc_v6.1_w_titles.json')
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    # HOTPOT(tokenizer, 'new_hotpot_train_v1.1.json', multi_label=True)
    # HOTPOT(tokenizer, 'new_hotpot_dev_distractor_v1.json', multi_label=True)
    # print('converting to features...')
    # convert_examples_to_features(examples, tokenizer, questions, contexts )
    # dataset = ROPES(tokenizer, 'dev-v1.0.json')
    # print(len(dataset[0]['input_ids']))
    # with open("data/hotpot/new_hotpot_dev_distractor_v1.json", 'r') as f:
    #     data = json.load(f)
    # sent_nums = []
    # for example in tqdm(data):
    #     context = example["context"]
    #     tokens, sent_num = [], 0
    #     for _, sents in context:
    #         if len(tokens) >= 512:
    #             break
    #         for sent in sents:
    #             tokens.extend(tokenizer.tokenize(sent)[:512 - len(tokens)])
    #             sent_num += 1
    #     sent_nums.append(sent_num)
    # print(np.mean(sent_nums))
