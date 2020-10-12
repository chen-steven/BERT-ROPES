from torch.utils.data import Dataset
import json
from tqdm import tqdm
import torch

ROPES_DATA_PATH = 'data/ropes/'
MAX_PARAGRAPH_LEN = 400

class ROPES(Dataset):
    def __init__(self, tokenizer, file_path, eval=False):
        self.tokenizer = tokenizer
        self.eval = eval
        contexts, questions, answers, qids = get_examples(file_path)
        self.qids = qids
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.encodings = tokenizer(contexts, questions, padding=True, truncation=True)
        self._update_start_end_idxs()

    def _update_start_end_idxs(self):
        starts = []
        ends = []
        question_encodings = self.tokenizer(self.questions)
        context_encodings = self.tokenizer(self.contexts)

        for i in tqdm(range(len(self.answers))):
            answer = self.answers[i]
            #tokens = self.tokenizer.convert_ids_to_tokens(self.encodings['input_ids'][i])
            q_idx = self.questions[i].find(answer)
            c_idx = self.contexts[i].find(answer)


            if q_idx != -1:
                assert(self.questions[i][q_idx:q_idx+len(answer)] == answer)
                y1 = question_encodings.char_to_token(i, q_idx)-1+len(context_encodings[i])
                y2 = question_encodings.char_to_token(i, q_idx+len(answer)-1)

                if not y2 and y1:
                    y2 = y1 + len(self.tokenizer.tokenize(answer)) - 1
                elif y1:
                    y2 += -1+len(context_encodings[i])

                y1 = min(y1, 512)
                y2 = min(y2, 512)
            elif c_idx != -1:
                y1 = self.encodings.char_to_token(i, c_idx)
                y2 = self.encodings.char_to_token(i, c_idx + len(answer) - 1)
                if not y1:
                    y1, y2 = 512, 512
                if not y2 and y1:
                    y2 = y1 + len(self.tokenizer.tokenize(answer)) - 1



            starts.append(y1)
            ends.append(y2)


        self.encodings['start_positions'] = starts
        self.encodings['end_positions'] = ends

    def __getitem__(self, idx):
        inputs = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.eval:
            inputs['id'] = self.qids[idx]
        return inputs

    def __len__(self):
        return len(self.answers)


def get_examples(file_path):
    contexts, questions, answers, qids = [], [], [], []
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
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)
                        qids.append(id)

    return contexts, questions, answers, qids

if __name__ == '__main__':
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    dataset = ROPES(tokenizer, 'train-v1.0.json')
    print(len(dataset[0]['input_ids']))


