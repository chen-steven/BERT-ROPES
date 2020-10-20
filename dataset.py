from torch.utils.data import Dataset
import json
from tqdm import tqdm
import torch
from utils import ROPESExample, convert_idx
from transformers import BasicTokenizer
from transformers.data.processors.squad import squad_convert_examples_to_features, SquadExample

ROPES_DATA_PATH = 'data/ropes/'
MAX_PARAGRAPH_LEN = 400
SWAP_STR = '***'
PUNCT = [',','?','.']

class ROPES(Dataset):
    def __init__(self, args, tokenizer, file_path, eval=False):
        self.args = args
        self.tokenizer = tokenizer
        self.eval = eval
        examples, questions, contexts = get_examples(file_path, augmented=args.use_augmented_examples)
        self.examples = examples


        self.encodings, self.start_multi_label, self.end_multi_label = convert_examples_to_features(examples,
                                                                    tokenizer, questions, contexts)
        print(len(self.start_multi_label), len(self.start_multi_label[0]))
        print(len(examples), len(questions), len(contexts))
    def __getitem__(self, idx):
        inputs = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.eval:
            inputs['id'] = self.examples[idx].qas_id

        if self.args.use_multi_labels and not self.eval:
            inputs['start_multi_label'] = self.start_multi_label[idx]
            inputs['end_multi_label'] = self.end_multi_label[idx]
        
        return inputs

    def __len__(self):
        return len(self.examples)

def convert_examples_to_features(examples, tokenizer, questions, contexts, max_seq_length=512, doc_stride=1):
    #TODO: also return features with ROPESFeatures object
    #features = []
    count = 0
    encodings = tokenizer(questions, contexts, padding=True, truncation=True)
    start_positions, end_positions = [], []
    start_multi_label, end_multi_label = [], []
    for i, example in tqdm(enumerate(examples)):
        start_labels, end_labels = [], []
        answer = example.answer
        question = example.question
        context = example.context
        q_idx = question.find(answer)
        c_idx = context.find(answer)
        if q_idx != -1:
            start_position = encodings.char_to_token(i, q_idx)
            end_position = encodings.char_to_token(i, q_idx+len(answer)-1)
            if start_position >= 512:
                start_position = 0
            if end_position >= 512:
                end_position = 0
            start_labels.append(start_position)
            end_labels.append(end_position)

        while q_idx != -1 and q_idx+len(answer) < len(question) and question[q_idx+len(answer)].isalpha():
           q_idx = question.find(answer, q_idx+1)
        while c_idx != -1 and c_idx+len(answer) < len(context) and context[c_idx+len(answer)].isalpha():
           c_idx = context.find(answer, c_idx+1)

        if c_idx != -1:
            question_tokens = tokenizer.tokenize(question)
            context_encoding = tokenizer(context)
            start_position = context_encoding.char_to_token(c_idx) + len(question_tokens) + 1
            end_position = context_encoding.char_to_token(c_idx+len(answer)-1) + len(question_tokens) + 1
            if start_position >= 512:
                start_position = 0
            if end_position >= 512:
                end_position = 0
            start_labels.append(start_position)
            end_labels.append(end_position)
        #elif q_idx != -1:
        #    start_position = encodings.char_to_token(i, q_idx)
        #    end_position = encodings.char_to_token(i, q_idx+len(answer)-1)
        if c_idx == -1 and q_idx == -1:
            print(question, answer)
            start_labels.append(0)
            end_labels.append(0)

        tmp = tokenizer.decode(encodings['input_ids'][i][start_position:end_position+1])
        if tmp != answer and start_position < 512 and end_position < 512:
            count += 1
            print(tmp, answer)

        if len(start_labels) == 1:
            start_labels.append(-1)
        if len(end_labels) == 1:
            end_labels.append(-1)

        start_positions.append(start_labels[0])
        end_positions.append(end_labels[0])
        start_multi_label.append(start_labels)
        end_multi_label.append(end_labels)
    encodings['start_positions'] = start_positions
    encodings['end_positions'] = end_positions
    print('Total incorrectly decoded examples:',count)
    return encodings, start_multi_label, end_multi_label

def create_augmented_example(question, answer):
    answer = answer.strip()
    processed_q = question.replace('?', ' ?').replace(',', ' ,')
    q_idx = processed_q.find(answer)
    if q_idx == -1:
        return None
    or_idx = processed_q.find(' or ')
    tokens = processed_q.split()
    spans = convert_idx(processed_q, tokens)
    answer_start, answer_end = q_idx, q_idx+len(answer)
    answer_span = []
    for idx, span in enumerate(spans):
        if not (answer_end <= span[0] or answer_start >= span[1]):
            answer_span.append(idx)
    start, end = answer_span[0], answer_span[-1]

    if or_idx != -1 and q_idx+len(answer) == or_idx:
        candidate_tokens = tokens[end+2:end+3+(end-start)]
        if candidate_tokens[0] in PUNCT:
            candidate_tokens.pop(0)
        if candidate_tokens[-1] in PUNCT:
            candidate_tokens.pop()
        candidate = ' '.join(candidate_tokens)
    elif or_idx != -1 and or_idx+len(' or ') == q_idx:
        candidate_tokens = tokens[start - (end-start) -2: start-1]
        if candidate_tokens[0] in PUNCT:
            candidate_tokens.pop(0)
        if candidate_tokens[-1] in PUNCT:
            candidate_tokens.pop()
        candidate = ' '.join(candidate_tokens)
    else:
        #print(question, answer, q_idx, or_idx, 'not found')
        return None

    tmp = question.replace(candidate, SWAP_STR)
    tmp = tmp.replace(answer, candidate)
    tmp = tmp.replace(SWAP_STR, answer)
    #print(tmp, candidate, answer)
    return tmp


def get_examples(file_path, augmented=False):
    examples, questions, contexts = [], [], []
    count, total = 0, 0
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

                            if question.find(' or ') != -1:
                                total += 1
                                if augmented:
                                    aug = create_augmented_example(question, answer)
                                    if aug:
                                        count += 1
                                        example = ROPESExample(id+'_1', aug, context, answer.strip(), aug.find(answer))
                                        examples.append(example)
                                        questions.append(aug)
                                        contexts.append(context)

                        else:
                            start_position = sit_idx + len(background) + 1
                        #example = SquadExample(id, question, context, answer, start_position, 'test')
                        example = ROPESExample(id, question, context, answer.strip(), start_position)
                        examples.append(example)
                        questions.append(question)
                        contexts.append(context)
    print(count, total)
    return examples, questions, contexts


if __name__ == '__main__':
    from transformers import BertTokenizerFast, AutoTokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    #tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    class O:
        pass
    args = O()
    args.use_augmented_examples=True
    args.use_multi_labels=True
    dataset = ROPES(args, tokenizer,'train-v1.0.json')
    print(dataset[0])


    #print('converting to features...')
    #convert_examples_to_features(examples, tokenizer, questions, contexts )

    #dataset = ROPES(tokenizer, 'dev-v1.0.json')
    #print(len(dataset[0]['input_ids']))


