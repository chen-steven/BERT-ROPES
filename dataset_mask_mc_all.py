from torch.utils.data import Dataset
import json
from tqdm import tqdm
import torch
import numpy as np
from datetime import datetime
from utils import ROPESExample
from transformers.data.processors.squad import squad_convert_examples_to_features, SquadExample
import nltk
nltk.data.path.append('./nltk_data')
nltk.download('punkt', download_dir='./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
ROPES_DATA_PATH = 'data/ropes/'
MAX_PARAGRAPH_LEN = 400


class ROPESMaskExample:
    def __init__(self, qas_id, question, context, answers, answer_subsents, answer_label):
        self.qas_id = qas_id
        self.question = question
        self.context = context
        self.answers = answers
        self.qas = [question + ' ' + answer for answer in answers]
        self.answer_subsents = answer_subsents
        self.answer_label = answer_label


class ROPES(Dataset):
    def __init__(self, tokenizer, file_path, answer_file_path, eval=False, find_mask="label"):
        self.tokenizer = tokenizer
        self.eval = eval
        examples, questions, qas, contexts, evaluation_file = get_examples(file_path, answer_file_path,
                                                                           find_mask=find_mask, eval=eval)
        self.evaluation_file = evaluation_file
        self.examples = examples
        self.encodings = convert_examples_to_features(examples, tokenizer, questions, qas, contexts)

    def __getitem__(self, idx):
        inputs = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.eval:
            inputs['id'] = self.examples[idx].qas_id
        return inputs

    def __len__(self):
        return len(self.examples)


def convert_examples_to_features(examples, tokenizer, qas1, qas2, contexts):
    # TODO: also return features with ROPESFeatures object
    # features = []
    all_qas_encodings = tokenizer(qas1 + qas2, contexts + contexts, padding=True, truncation=True)
    qas_encodings = [{}, {}]
    for key in all_qas_encodings:
        length = len(all_qas_encodings[key]) // 2
        qas_encodings[0][key] = all_qas_encodings[key][:length]
        qas_encodings[1][key] = all_qas_encodings[key][length:]
    qas_mask_labels, qas_mask_inputs, answer_labels = [[], []], [[], []], []
    for i, example in tqdm(enumerate(examples)):
        qas = example.qas
        context = example.context
        answer_subsents = example.answer_subsents
        context_encoding = tokenizer(context)
        answer_labels.append(example.answer_label)
        for j in range(2):
            qa_tokens = tokenizer.tokenize(qas[j])
            encodings = qas_encodings[j]

            mask_label = [-100] * len(encodings["input_ids"][i])
            mask_input = encodings["input_ids"][i]

            for answer_subsent in answer_subsents:
                if answer_subsent is not None:
                    ac_idx = context.find(answer_subsent)
                    ac_start = context_encoding.char_to_token(ac_idx)
                    ac_end = context_encoding.char_to_token(ac_idx + len(answer_subsent) - 1)

                    as_start = ac_start + len(qa_tokens) + 1
                    as_end = ac_end + len(qa_tokens) + 1
                    mask_label[as_start:as_end+1] = encodings["input_ids"][i][as_start:as_end+1]
                    mask_input[as_start:as_end+1] = [tokenizer.encode("[MASK]")[1]] * (as_end + 1 - as_start)

            assert len(mask_label) == len(mask_input)
            qas_mask_labels[j].append(mask_label)
            qas_mask_inputs[j].append(mask_input)

    encodings = {}
    for i in range(2):
        for key in qas_encodings[i]:
            encodings[f"{key}{i}"] = qas_encodings[i][key]
        encodings[f'mask_inputs{i}'] = qas_mask_inputs[i]
        encodings[f'mask_labels{i}'] = qas_mask_labels[i]
    encodings["answer_labels"] = answer_labels
    return encodings


def get_examples(file_path, answer_file_path, find_mask="label", eval=False):
    examples, qas1, qas2, contexts = [], [], [], []
    evaluate_answers = {}
    with open(f'{ROPES_DATA_PATH}{file_path}', 'r', encoding='utf-8') as f, \
            open(f'{ROPES_DATA_PATH}{answer_file_path}', 'r', encoding='utf-8') as fa:
        data = json.load(f)
        data_answers = json.load(fa)
        for article in tqdm(data):
            id = article["id"]
            question = article["question"]
            answers, answer_label = data_answers[id]
            if answer_label == -1:
                if not eval:
                    answer = article["answer"].strip()
                    answers[0], answer_label = answer, 0
                else:
                    answer_label = 0
            # shuffle answers
            np.random.seed(42)
            if np.random.random() > 0.5:
                answers = [answers[1], answers[0]]
                answer_label = 1 if answer_label == 0 else 0

            context_sent = article["context_sent"]
            situation_length = len(sent_tokenize(article["situation"]))

            answer_subsents = []
            for answer in answers:
                # find the answer sentence
                answer_subsent = None
                if "fact_labels" in article:
                    for fact_label in article["fact_labels"]:
                        if fact_label == len(context_sent):
                            break
                        sent = context_sent[fact_label]
                        if sent.find(answer) != -1:
                            subsents = sent.split(',')
                            for i, subsent in enumerate(subsents):
                                if subsent.find(answer) != -1:
                                    answer_subsent = subsent
                                    if len(answer) / len(answer_subsent) > 0.4 and i + 1 < len(subsents):
                                        answer_subsent = answer_subsent + ',' + subsents[i+1]
                                        break
                            break
                if find_mask == "label1" or find_mask == "label2":
                    if answer_subsent is None:
                        situation_sent = context_sent[:situation_length]
                        answer_sent = []
                        for sent in situation_sent:
                            if sent.find(answer) != -1:
                                answer_sent.append(sent)
                        if len(answer_sent) > 0:
                            np.random.shuffle(answer_sent)
                            sent = answer_sent[0]
                            subsents = sent.split(',')
                            for i, subsent in enumerate(subsents):
                                if subsent.find(answer) != -1:
                                    answer_subsent = subsent
                                    if len(answer) / len(answer_subsent) > 0.4 and i + 1 < len(subsents):
                                        answer_subsent = answer_subsent + ',' + subsents[i + 1]
                                        break
                if find_mask == "label2":
                    if answer_subsent is None:
                        question_tokens = [w for w in word_tokenize(question) if not w in stop_words]
                        best_sent, best_overlap = None, -1
                        situation_sent = context_sent[:situation_length]
                        for sent in situation_sent:
                            sent_tokens = [w for w in word_tokenize(sent) if not w in stop_words]
                            overlap = len(set(sent_tokens) & set(question_tokens)) / len(question_tokens)
                            if overlap > best_overlap:
                                best_sent, best_overlap = sent, overlap
                        subsents = sent.split(',')
                        for i, subsent in enumerate(subsents):
                            answer_subsent = subsent
                            if len(answer_subsent) < 20 and i + 1 < len(subsents):
                                answer_subsent = answer_subsent + ',' + subsents[i + 1]
                                break
                if answer_subsent is not None:
                    answer_subsent = answer_subsent.strip()
                answer_subsents.append(answer_subsent)

            if None in answer_subsents:
                answer_subsents = [None, None]

            context = ' '.join(context_sent)
            example = ROPESMaskExample(id, question, context, answers, answer_subsents, answer_label)
            examples.append(example)
            qas1.append(question + ' ' + answers[0])
            qas2.append(question + ' ' + answers[1])
            contexts.append(context)
            evaluate_answers[id] = answers

    evaluate_answer_file_path = f"{ROPES_DATA_PATH}evaluate_{find_mask}_{datetime.now().strftime('%m_%d_%y_%H:%M:%S')}_{answer_file_path}"
    with open(evaluate_answer_file_path, 'w') as f:
        json.dump(evaluate_answers, f)

    return examples, qas1, qas2, contexts, evaluate_answer_file_path


if __name__ == '__main__':
    from transformers import BertTokenizerFast, AutoTokenizer

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', cache_dir="./cache")
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    ROPES(tokenizer, 'ropes_no_coref_dev_combined_examples.json', 'dev_processed_candidate_answers.json',
          find_mask="label2")
    # print('converting to features...')
    # convert_examples_to_features(examples, tokenizer, questions, contexts )

    # dataset = ROPES(tokenizer, 'dev-v1.0.json')
    # print(len(dataset[0]['input_ids']))
