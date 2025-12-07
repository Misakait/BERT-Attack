import warnings
import os
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertConfig, BertTokenizer
from transformers import BertForSequenceClassification, BertForMaskedLM
import copy
import argparse
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

filter_words = []
filter_words = set(filter_words)


def get_sim_embed(embed_path, sim_path):
    id2word = {}
    word2id = {}

    with open(embed_path, 'r', encoding='utf-8') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in id2word:
                id2word[len(id2word)] = word
                word2id[word] = len(id2word) - 1

    cos_sim = np.load(sim_path)
    return cos_sim, word2id, id2word


def get_data_cls(data_path):
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

    original_len = len(df)
    df.dropna(subset=['is_fraud'], inplace=True)
    print(f"Dropped {original_len - len(df)} invalid rows.")

    features = []
    for index, row in df.iterrows():
        try:
            label_raw = row['is_fraud']
            if isinstance(label_raw, bool):
                label = 1 if label_raw else 0
            else:
                label = 1 if str(label_raw).lower() == 'true' else 0
        except:
            label = 0

        raw_text = str(row['specific_dialogue_content'])
        seq = raw_text.replace('音频内容：', '').replace('\n', ' ').replace('\r', '')
        seq = ' '.join(seq.split())

        features.append([seq, label])

    print(f"Successfully loaded {len(features)} samples.")
    return features


class Feature(object):
    def __init__(self, seq_a, label):
        self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []


def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '').lower()

    # 使用 BERT tokenizer 去分割中文语句
    words = tokenizer.tokenize(seq)

    sub_words = []
    keys = []
    index = 0

    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys


def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    return masked_words


def get_important_scores(words, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length):
    masked_words = _get_masked(words)
    texts = [' '.join(words) for words in masked_words]

    all_input_ids = []
    all_masks = []
    all_segs = []
    for text in texts:
        inputs = tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=max_length, truncation=True)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + (padding_length * [0])
        token_type_ids = token_type_ids + (padding_length * [0])
        attention_mask = attention_mask + (padding_length * [0])
        all_input_ids.append(input_ids)
        all_masks.append(attention_mask)
        all_segs.append(token_type_ids)
    seqs = torch.tensor(all_input_ids, dtype=torch.long)
    masks = torch.tensor(all_masks, dtype=torch.long)
    segs = torch.tensor(all_segs, dtype=torch.long)
    seqs = seqs.to('cuda')

    eval_data = TensorDataset(seqs)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    leave_1_probs = []

    with torch.no_grad():
        for batch in eval_dataloader:
            masked_input, = batch
            leave_1_prob_batch = tgt_model(masked_input)[0]
            leave_1_probs.append(leave_1_prob_batch)

    leave_1_probs = torch.cat(leave_1_probs, dim=0)
    leave_1_probs = torch.softmax(leave_1_probs, -1)
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

    import_scores = (orig_prob
                     - leave_1_probs[:, orig_label]
                     +
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()

    return import_scores


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    words = []
    sub_len, k = substitutes.size()

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    substitutes = substitutes[0:12, 0:4]
    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    c_loss = nn.CrossEntropyLoss(reduction='none')
    all_substitutes = torch.tensor(all_substitutes)
    all_substitutes = all_substitutes[:24].to('cuda')
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words


def attack(feature, tgt_model, mlm_model, tokenizer, k, batch_size, max_length=512, cos_mat=None, w2i={}, i2w={},
           use_bpe=1, threshold_pred_score=0.3):
    words, sub_words, keys = _tokenize(feature.seq, tokenizer)

    inputs = tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length, truncation=True)
    input_ids, token_type_ids = torch.tensor(inputs["input_ids"]), torch.tensor(inputs["token_type_ids"])
    attention_mask = torch.tensor([1] * len(input_ids))

    orig_probs = tgt_model(input_ids.unsqueeze(0).to('cuda'),
                           attention_mask.unsqueeze(0).to('cuda'),
                           token_type_ids.unsqueeze(0).to('cuda')
                           )[0].squeeze()
    orig_probs = torch.softmax(orig_probs, -1)
    orig_label = torch.argmax(orig_probs)
    current_prob = orig_probs.max()

    if orig_label != feature.label:
        feature.success = 3
        return feature

    sub_words_tensor = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
    input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words_tensor)])

    # Get predictions from MLM
    word_predictions = mlm_model(input_ids_.to('cuda'))[0].squeeze()
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)

    # Align predictions with words
    word_predictions = word_predictions[1:len(sub_words) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

    # Calculate vulnerability
    important_scores = get_important_scores(words, tgt_model, current_prob, orig_label, orig_probs,
                                            tokenizer, batch_size, max_length)
    feature.query += int(len(words))
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

    final_words = copy.deepcopy(words)

    for top_index in list_of_index:
        if feature.change > int(0.4 * (len(words))):
            feature.success = 1  # exceed
            return feature

        tgt_word = words[top_index[0]]
        if tgt_word in filter_words:
            continue

        # Check bounds
        if top_index[0] >= len(word_predictions):
            continue

        substitutes = word_predictions[top_index[0]:top_index[0] + 1]  # Take the row for this word
        word_pred_scores = word_pred_scores_all[top_index[0]:top_index[0] + 1]

        substitutes = get_substitues(substitutes, tokenizer, mlm_model, use_bpe, word_pred_scores, threshold_pred_score)

        most_gap = 0.0
        candidate = None

        for substitute_ in substitutes:
            substitute = substitute_

            if substitute == tgt_word:
                continue
            if '##' in substitute:
                continue
            if substitute in filter_words:
                continue
            # [Modified] Skip Sim Matrix check if it's None (User should set use_sim_mat=0)
            if cos_mat is not None and substitute in w2i and tgt_word in w2i:
                if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                    continue

            temp_replace = final_words[:]
            temp_replace[top_index[0]] = substitute

            temp_text = tokenizer.convert_tokens_to_string(temp_replace)

            inputs = tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length,
                                           truncation=True)
            input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to('cuda')

            temp_prob = tgt_model(input_ids)[0].squeeze()
            feature.query += 1
            temp_prob = torch.softmax(temp_prob, -1)
            temp_label = torch.argmax(temp_prob)

            if temp_label != orig_label:
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append([top_index[0], substitute, tgt_word])
                feature.final_adverse = temp_text
                feature.success = 4
                return feature
            else:
                label_prob = temp_prob[orig_label]
                gap = current_prob - label_prob
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            feature.change += 1
            feature.changes.append([top_index[0], candidate, tgt_word])
            current_prob = current_prob - most_gap
            final_words[top_index[0]] = candidate

    feature.final_adverse = tokenizer.convert_tokens_to_string(final_words)
    feature.success = 2
    return feature


def evaluate(features):
    acc = 0
    origin_success = 0
    total = 0
    total_q = 0
    total_change = 0
    total_word = 0

    for feat in features:
        if feat.success > 2:
            acc += 1
            total_q += feat.query
            total_change += feat.change
            total_word += len(feat.seq)

            if feat.success == 3:
                origin_success += 1

        total += 1

    if total == 0:
        print("No samples evaluated.")
        return

    suc = float(acc / total)
    query = float(total_q / acc) if acc > 0 else 0
    change_rate = float(total_change / total_word) if total_word > 0 else 0

    origin_acc = 1 - origin_success / total
    after_atk = 1 - suc

    print('Results:')
    print(f'Original Accuracy: {origin_acc:.4f}')
    print(f'After-Attack Accuracy: {after_atk:.4f}')
    print(f'Avg Queries: {query:.4f}')
    print(f'Avg Change Rate: {change_rate:.4f}')


def dump_features(features, output):
    outputs = []
    for feature in features:
        outputs.append({'label': int(feature.label),
                        'success': int(feature.success),
                        'change': int(feature.change),
                        'num_word': len(feature.seq),
                        'query': int(feature.query),
                        'changes': feature.changes,
                        'seq_a': feature.seq,
                        'adv': feature.final_adverse,
                        })
    json.dump(outputs, open(output, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'Finished dumping logs to {output}')


def run_attack():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mlm_path", type=str, required=True, help="BERT-base-chinese")
    parser.add_argument("--tgt_path", type=str, required=True, help="Fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="attack_log.json")
    parser.add_argument("--use_sim_mat", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1000)
    parser.add_argument("--num_label", type=int, default=2)
    parser.add_argument("--use_bpe", type=int, default=1)
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--threshold_pred_score", type=float, default=0.0)

    args = parser.parse_args()

    # Load Models
    print('Loading models...')
    tokenizer_tgt = BertTokenizer.from_pretrained(args.tgt_path)

    # Use BERT-base-chinese for MLM if not specified fully (but user should pass it)
    tokenizer_mlm = BertTokenizer.from_pretrained(args.mlm_path)

    config_atk = BertConfig.from_pretrained(args.mlm_path)
    mlm_model = BertForMaskedLM.from_pretrained(args.mlm_path, config=config_atk)
    mlm_model.to('cuda')

    config_tgt = BertConfig.from_pretrained(args.tgt_path, num_labels=args.num_label)
    tgt_model = BertForSequenceClassification.from_pretrained(args.tgt_path, config=config_tgt)
    tgt_model.to('cuda')
    tgt_model.eval()

    features = get_data_cls(args.data_path)

    #  Sim Matrix (Skipped)
    cos_mat, w2i, i2w = None, {}, {}
    if args.use_sim_mat == 1:
        print("Warning: Sim matrix logic is not adapted for Chinese. Setting to None.")

    features_output = []

    print(f'Starting attack process (Samples {args.start} to {args.end})...')

    # Slice the data safely
    data_slice = features[args.start: min(args.end, len(features))]

    with torch.no_grad():
        for index, feature in enumerate(data_slice):
            seq_a, label = feature
            feat = Feature(seq_a, label)

            print(f'\rProcessing: {index + args.start}/{min(args.end, len(features))} | ', end='')

            feat = attack(feat, tgt_model, mlm_model, tokenizer_tgt, args.k,
                          batch_size=32, max_length=512,
                          cos_mat=cos_mat, w2i=w2i, i2w=i2w, use_bpe=args.use_bpe,
                          threshold_pred_score=args.threshold_pred_score)

            if feat.success > 2:
                print('Success', end='')
            else:
                print('Failed', end='')

            features_output.append(feat)

    print('\nEvaluating results...')
    evaluate(features_output)
    dump_features(features_output, args.output_dir)


if __name__ == '__main__':
    run_attack()