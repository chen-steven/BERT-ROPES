from torch.utils.data import Dataset
import json
from tqdm import tqdm
import torch
from utils import ROPESExample
from transformers.data.processors.squad import squad_convert_examples_to_features, SquadExample

ROPES_DATA_PATH = 'data/ropes/'
MAX_PARAGRAPH_LEN = 400

class ROPES(Dataset):
    def __init__(self, tokenizer, file_path, eval=False):
        self.tokenizer = tokenizer
        self.eval = eval
        examples, questions, contexts = get_examples(file_path)
        self.examples = examples
        self.encodings, self.start_labels, self.end_labels = convert_examples_to_features(examples, tokenizer, questions, contexts)

    def __getitem__(self, idx):
        inputs = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.eval:
            inputs['id'] = self.examples[idx].qas_id
        return inputs, torch.tensor(self.start_labels[idx]), torch.tensor(self.end_labels[idx])

    def __len__(self):
        return len(self.examples)

def convert_examples_to_features(examples, tokenizer, questions, contexts, max_seq_length=512, doc_stride=1):
    #TODO: also return features with ROPESFeatures object
    #features = []
    encodings = tokenizer(questions, contexts, padding=True, truncation=True)
    start_positions, end_positions = [], []
    start_labels, end_labels = [], []
    for i, example in tqdm(enumerate(examples)):
        answer = example.answer
        question = example.question
        context = example.context
        q_idx = question.find(answer)
        c_idx = context.find(answer)

        cur_start_label, cur_end_label = [0]*len(encodings['input_ids'][0]), [0]*len(encodings['input_ids'][0])
        if q_idx != -1:
            start_position = encodings.char_to_token(i, q_idx)
            end_position = encodings.char_to_token(i, q_idx+len(answer)-1)
            if start_position < len(cur_start_label): cur_start_label[start_position] = 1
            if end_position < len(cur_end_label): cur_end_label[end_position] = 1
#        while q_idx != -1 and q_idx+len(answer) < len(question) and question[q_idx+len(answer)].isalpha():
#            q_idx = question.find(answer, q_idx+1)
#        while c_idx != -1 and c_idx+len(answer) < len(context) and context[c_idx+len(answer)].isalpha():
#            c_idx = context.find(answer, c_idx+1)
        elif c_idx != -1:
            question_tokens = tokenizer.tokenize(question)
            context_encoding = tokenizer(context)
            start_position = context_encoding.char_to_token(c_idx) + len(question_tokens) + 1
            end_position = context_encoding.char_to_token(c_idx+len(answer)-1) + len(question_tokens) + 1
            if start_position < len(cur_start_label): cur_start_label[start_position] = 1
            if end_position < len(cur_end_label): cur_end_label[end_position] = 1
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
        start_labels.append(cur_start_label)
        end_labels.append(cur_end_label)
    encodings['start_positions'] = start_positions
    encodings['end_positions'] = end_positions
    return encodings, start_labels, end_labels


def get_examples(file_path):
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


