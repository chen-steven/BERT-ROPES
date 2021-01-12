from torch.utils.data import Dataset
import json
from tqdm import tqdm
import torch
from utils import ROPESExample
from transformers.data.processors.squad import squad_convert_examples_to_features, SquadExample
import numpy as np

ROPES_DATA_PATH = 'data/ropes/'
MAX_PARAGRAPH_LEN = 400

class ROPES_MC(Dataset):
    def __init__(self, tokenizer, file_path, split='train'):
        self.tokenizer = tokenizer
        examples = get_mc_examples(file_path, split)
        self.features, self.qa_labels, self.ids = convert_mc_examples_to_features(examples, tokenizer)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        encodings = {key: torch.tensor(val) for key, val in self.features[idx].items()}
        qa_labels = torch.tensor(self.qa_labels[idx])
        ids = self.ids[idx]
        return encodings, qa_labels, ids

class ROPES(Dataset):
    def __init__(self, tokenizer, file_path, eval=False):
        self.tokenizer = tokenizer
        self.eval = eval
        examples, questions, contexts = get_examples(file_path)
        self.examples = examples
        self.encodings = convert_examples_to_features(examples, tokenizer, questions, contexts)

    def __getitem__(self, idx):
        inputs = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.eval:
            inputs['id'] = self.examples[idx].qas_id
        return inputs

    def __len__(self):
        return len(self.examples)

def convert_examples_to_features(examples, tokenizer, questions, contexts, max_seq_length=512, doc_stride=1):
    #TODO: also return features with ROPESFeatures object
    #features = []
    encodings = tokenizer(questions, contexts, padding=True, truncation=True)
    start_positions, end_positions = [], []
    for i, example in tqdm(enumerate(examples)):
        answer = example.answer
        question = example.question
        context = example.context
        q_idx = question.find(answer)
        c_idx = context.find(answer)
        if q_idx != -1:
            start_position = encodings.char_to_token(i, q_idx)
            end_position = encodings.char_to_token(i, q_idx+len(answer)-1)
#        while q_idx != -1 and q_idx+len(answer) < len(question) and question[q_idx+len(answer)].isalpha():
#            q_idx = question.find(answer, q_idx+1)
#        while c_idx != -1 and c_idx+len(answer) < len(context) and context[c_idx+len(answer)].isalpha():
#            c_idx = context.find(answer, c_idx+1)
        elif c_idx != -1:
            question_tokens = tokenizer.tokenize(question)
            context_encoding = tokenizer(context)
            start_position = context_encoding.char_to_token(c_idx) + len(question_tokens) + 1
            end_position = context_encoding.char_to_token(c_idx+len(answer)-1) + len(question_tokens) + 1
        #elif q_idx != -1:
        #    start_position = encodings.char_to_token(i, q_idx)
        #    end_position = encodings.char_to_token(i, q_idx+len(answer)-1)
        else:
            start_position = 0
            end_position = 0
        tmp = tokenizer.decode(encodings['input_ids'][i][start_position:end_position+1])
        if tmp != answer and start_position < 512 and end_position < 512:
            print(tmp, answer)
        if start_position >= 512:
            start_position = 0
        if end_position >= 512:
            end_position = 0

        start_positions.append(start_position)
        end_positions.append(end_position)
    encodings['start_positions'] = start_positions
    encodings['end_positions'] = end_positions
    return encodings

def convert_mc_examples_to_features(examples, tokenizer):
    features = []
    qa_labels = []
    ids = []
    for ex in tqdm(examples):
        ids.append(ex.qas_id)
        question = ex.question
        context = ex.context

#        feature_entry = []
        qa_labels.append(ex.answer_label)

        qa_inputs = []
        for ans in ex.candidate_answers:
            qa_input_text = question + ' [SEP] ' + ans + ' [SEP] ' + context
            qa_inputs.append(qa_input_text)

        qa_input = tokenizer(qa_inputs, padding="max_length", max_length=512, truncation=True)
            
 #       feature_entry.append(dict(**qa_input))

        features.append(dict(**qa_input))

    return features, qa_labels, ids
#    data = dict(features=features, qa_labels=qa_labels, ids=ids)
    

def get_mc_examples(file_path, split):
    candidate_answers = json.load(open(f'{split}_candidate_answers.json', 'r'))
    examples, questions, contexts = get_examples(file_path)
    bad_choices = ['', '[SEP]']

    processed_candidates = {}
    total = count = 0
    for example in examples:

        qid = example.qas_id
        answer = example.answer
        candidate_answer = ""

        if split=='dev' or split=='contrast':
            example.candidate_answers = candidate_answers[qid][1:]
            answer_label = -1
            for i, ans in enumerate(example.candidate_answers):
                if ans.lower() == answer.lower():
                    answer_label = i
                    count += 1
                    break
            total += 1
            example.answer_label = answer_label
            processed_candidates[qid] = (example.candidate_answers, answer_label)
            continue
            
        for ans in candidate_answers[qid]:
            if ans in bad_choices or answer.lower() == ans.lower(): continue

            idx = ans.lower().find(answer.lower())
            if idx == -1:
                candidate_answer = ans
                break
            elif len(ans) > len(answer):
                l1 = len(ans)-(idx+len(answer))
                l2 = idx
                if l2 > l1:
                    candidate_answer = ans[:idx]
                else:
                    candidate_answer = ans[idx+len(answer):]
                    
        if candidate_answer == '':
            print(candidate_answers[qid])
        if np.random.random() < 0.5:
            example.candidate_answers = [answer, candidate_answer]
            example.answer_label = 0
            processed_candidates[qid] = ([answer, candidate_answer], 0)
        else:
            example.candidate_answers = [candidate_answer, answer]
            example.answer_label = 1
            processed_candidates[qid] = ([candidate_answer, answer], 1)

    print(count, total)
    json.dump(processed_candidates, open(f'{split}_processed_candidate_answers.json', 'w'))
    return examples
   

def get_examples(file_path):
    print("Getting examples...")
    examples, questions, contexts = [], [], []
    with open(f'{ROPES_DATA_PATH}{file_path}', 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = data['data']
        for article in tqdm(data):
            for para in article['paragraphs']:
                background = para['background']
                situation = para['situation']
                context = background + ' ' + situation
                for qa in para['qas']:
                    id = qa['id']
                    question = qa['question']
                    for ans in qa['answers']:
                        answer = ans['text']
                        sit_idx = situation.find(answer)
                        q_idx = question.find(answer)
                        if q_idx != -1:
                            start_position = q_idx
                        else:
                            start_position = sit_idx + len(background) + 1
                        #example = SquadExample(id, question, context, answer, start_position, 'test')
                        example = ROPESExample(id, question, context, answer.strip(), start_position)
                        examples.append(example)
                        questions.append(question)
                        contexts.append(context)
    return examples, questions, contexts


if __name__ == '__main__':
    from transformers import BertTokenizerFast, AutoTokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    #tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    ROPES(tokenizer,'dev-v1.0.json')
    #print('converting to features...')
    #convert_examples_to_features(examples, tokenizer, questions, contexts )

    #dataset = ROPES(tokenizer, 'dev-v1.0.json')
    #print(len(dataset[0]['input_ids']))


