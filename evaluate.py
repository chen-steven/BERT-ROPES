import collections
import re
import string
import json
import torch
from tqdm import tqdm

class ROPESResult:
    def __init__(self, qid, start_logits, end_logits, input_ids):
        self.qid = qid
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.input_ids = input_ids

def _get_best_indexes(logits, n_best_size=512):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores1(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

    return exact_scores, f1_scores
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def get_raw_scores(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    #print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    compute_exact, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    compute_f1, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

def convert_to_predictions(predictions, tokenizer):
    ans_predictions = {}
    for qid, res in tqdm(predictions.items()):
        start_indexes = _get_best_indexes(res.start_logits)
        end_indexes = _get_best_indexes(res.end_logits)
        seq_len = torch.sum(res.input_ids != 0)
        s, e = 0,0
        for start_index in start_indexes:
            found = False
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= seq_len:
                    continue
                if end_index >= seq_len:
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > 5:
                    continue
                s = start_index
                e = end_index
                found=True
                break
            if found:
                break
        #start_idx = start_indexes[0]
        #end_idx = end_indexes[0]
        ans = ""
        if s < 512 and e < 512 and s < e:
            ans = tokenizer.decode(res.input_ids.tolist()[s:e+1])
        ans_predictions[qid] = ans

    with open('predictions.out', 'w') as f:
        json.dump(ans_predictions, f)
    return ans_predictions


def main(predictions, tokenizer, contrast=False):
    with open('data/ropes/ropes_contrast_set_original_032820.json' if contrast else 'data/ropes/dev-v1.0.json', 'r') as f:
        data = json.load(f)

    #ans_predictions = convert_to_predictions(predictions, tokenizer)
    return get_raw_scores(data['data'], predictions)
