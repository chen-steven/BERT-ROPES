from torch.utils.data import Dataset
import json
from tqdm import tqdm
import torch
import numpy as np
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
    def __init__(self, qas_id, question, context, answer, answer_subsent, start_position):
        self.qas_id = qas_id
        self.question = question
        self.context = context
        self.answer = answer
        self.qa = question + ' ' + answer
        self.answer_subsent = answer_subsent
        self.start_character = start_position
        self.end_position = start_position + len(answer)


class ROPES(Dataset):
    def __init__(self, tokenizer, file_path, eval=False, find_mask="label"):
        self.tokenizer = tokenizer
        self.eval = eval
        examples, questions, qas, contexts = get_examples(file_path, find_mask=find_mask)
        self.examples = examples
        self.encodings = convert_examples_to_features(examples, tokenizer, questions, qas, contexts)

    def __getitem__(self, idx):
        inputs = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.eval:
            inputs['id'] = self.examples[idx].qas_id
        return inputs

    def __len__(self):
        return len(self.examples)


def convert_examples_to_features(examples, tokenizer, questions, qas, contexts,
                                 max_seq_length=512, doc_stride=1, multi_label=False):
    # TODO: also return features with ROPESFeatures object
    # features = []
    encodings = tokenizer(questions, contexts, padding=True, truncation=True)
    mask_encodings = tokenizer(qas, contexts, padding=True, truncation=True)
    start_positions, end_positions = [], []
    start_labels, end_labels = [], []
    mask_inputs, mask_labels = [], []
    for i, example in tqdm(enumerate(examples)):
        answer = example.answer
        question = example.question
        qa = example.qa
        context = example.context
        answer_subsent = example.answer_subsent
        q_idx = question.find(answer)
        c_idx = context.find(answer)
        question_tokens = tokenizer.tokenize(question)
        qa_tokens = tokenizer.tokenize(qa)
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
        if tmp != answer and start_position < 512 and end_position < 512:
            print(tmp, answer)
        if start_position >= 512:
            start_position = 0
        if end_position >= 512:
            end_position = 0
        start_positions.append(start_position)
        end_positions.append(end_position)

        if answer_subsent is not None:
            ac_idx = context.find(answer_subsent)
            as_start = context_encoding.char_to_token(ac_idx) + len(qa_tokens) + 1
            as_end = context_encoding.char_to_token(ac_idx + len(answer_subsent) - 1) + len(qa_tokens) + 1
            mask_label = [-100] * as_start + mask_encodings["input_ids"][i][as_start:as_end+1] + \
                         [-100] * (len(mask_encodings["input_ids"][i]) - as_end - 1)
            mask_labels.append(mask_label)
            mask_input = mask_encodings["input_ids"][i][:as_start] + \
                         [tokenizer.encode("[MASK]")[1]] * (as_end + 1 - as_start) + \
                         mask_encodings["input_ids"][i][as_end+1:]
            mask_inputs.append(mask_input)
        else:
            mask_label = [-100] * len(mask_encodings["input_ids"][i])
            mask_labels.append(mask_label)
            mask_input = mask_encodings["input_ids"][i]
            mask_inputs.append(mask_input)
        assert len(mask_label) == len(mask_input) == len(mask_encodings["input_ids"][i])

    encodings['start_positions'] = start_positions
    encodings['end_positions'] = end_positions
    encodings['mask_inputs'] = mask_inputs
    encodings['mask_labels'] = mask_labels
    encodings['mask_type_ids'] = mask_encodings["token_type_ids"]
    encodings['mask_attention_mask'] = mask_encodings["attention_mask"]
    return encodings


def get_examples(file_path, find_mask="label"):
    examples, questions, qas, contexts = [], [], [], []
    with open(f'{ROPES_DATA_PATH}{file_path}', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for article in tqdm(data):
            id = article["id"]
            question = article["question"]
            answer = article["answer"].strip()
            context_sent = article["context_sent"]
            situation_length = len(sent_tokenize(article["situation"]))

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
            context = ' '.join(context_sent)
            c_idx = context.find(answer)
            q_idx = question.find(answer)
            if q_idx != -1:
                start_position = q_idx
            else:
                start_position = c_idx
            example = ROPESMaskExample(id, question, context, answer, answer_subsent, start_position)
            examples.append(example)
            questions.append(question)
            qas.append(question + ' ' + answer)
            contexts.append(context)
    return examples, questions, qas, contexts


if __name__ == '__main__':
    from transformers import BertTokenizerFast, AutoTokenizer

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', cache_dir="./cache")
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    ROPES(tokenizer, 'ropes_no_coref_dev_combined_examples.json')
    # print('converting to features...')
    # convert_examples_to_features(examples, tokenizer, questions, contexts )

    # dataset = ROPES(tokenizer, 'dev-v1.0.json')
    # print(len(dataset[0]['input_ids']))
