import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def combine_and_pad_tokens(questions_tokens, answers_tokens, eos_id, max_length):
    """
    Combine and pad tokenized questions and answers.
    """
    combined_tokens = [q_tokens + a_tokens for q_tokens, a_tokens in zip(questions_tokens, answers_tokens)]
    padded_tokens = [tokens + [eos_id] * (max_length - len(tokens)) for tokens in combined_tokens]
    input_ids = torch.tensor(padded_tokens)
    attention_mask = torch.tensor([[1 if token != eos_id else 0 for token in seq] for seq in padded_tokens])
    return input_ids, attention_mask

def calculate_log_probability(model, tokenizer, questions, answers, eos_id, max_length=128):
    """
    Calculate the log probabilities of the answers given the questions.
    """
    if isinstance(questions, str):
        questions = [questions]
    if isinstance(answers, str):
        answers = [answers]

    questions_tokens = [tokenizer.encode(question, add_special_tokens=False) for question in questions]
    answers_tokens = [tokenizer.encode(answer, add_special_tokens=False) for answer in answers]
    for answer_tokens in answers_tokens:
        answer_tokens.append(eos_id)
    answers_tokens = [answer_tokens[:max_length] for answer_tokens in answers_tokens]

    gen_max_length = max(len(q_tokens) + len(a_tokens) for q_tokens, a_tokens in zip(questions_tokens, answers_tokens))

    input_ids, attention_mask = combine_and_pad_tokens(questions_tokens, answers_tokens, eos_id, gen_max_length)

    with torch.no_grad():
        outputs = model(input_ids=input_ids.to('cuda'), attention_mask=attention_mask.to('cuda'), labels=input_ids.to('cuda'))
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        total_log_probs = []
        for idx, answer_tokens in enumerate(answers_tokens):
            start_index = len(questions_tokens[idx])
            end_index = start_index + len(answer_tokens)

            log_prob = sum(log_probs[idx][index - 1][token].item() for index, token in enumerate(answer_tokens, start=start_index))
            total_log_probs.append(log_prob)

    return total_log_probs

def main(model_lists, sft_models_dir, output_file):
    """
    Main function to load data, generate adversarial examples, and save the results.
    """
    with open(output_file, "r") as json_file:
        data = json.load(json_file)

    for sft_step in tqdm(model_lists, desc='Policy models', position=0):
        sft = f'sft_{sft_step}'
        model_name = f"{sft_models_dir}/{sft}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
        tokenizer.pad_token = tokenizer.eos_token
        eos_id = tokenizer.eos_token_id

        for i in tqdm(range(len(data)), desc='RL Questions', leave=False, position=1):
            rl_question = data[i]
            question = rl_question['question']
            question = ["Question: " + question + "\n\nAnswer: "]
            answers = [rl_question['gold_answer'][0]]
            for worse_answer in rl_question['worse_answers']:
                answers.append(worse_answer[0])

            log_probs = calculate_log_probability(model, tokenizer, question * len(answers), answers, eos_id)

            if 'log_prob' not in data[i]['gold_answer'][1]['PM']:
                data[i]['gold_answer'][1]['PM']['log_prob'] = {}
            data[i]['gold_answer'][1]['PM']['log_prob'][sft] = log_probs[0]

            for j in range(len(data[i]['worse_answers'])):
                if 'PM' not in data[i]['worse_answers'][j][1]:
                    data[i]['worse_answers'][j][1]['PM'] = {}
                if 'log_prob' not in data[i]['worse_answers'][j][1]['PM']:
                    data[i]['worse_answers'][j][1]['PM']['log_prob'] = {}
                data[i]['worse_answers'][j][1]['PM']['log_prob'][sft] = log_probs[j + 1]

    with open(output_file, "w") as json_file:
        json.dump(data, json_file)
        print(f'Dumped JSON to {output_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SEAM data with policy models.")
    parser.add_argument('--model_lists', nargs='+', required=True, help='List of model steps to process.')
    parser.add_argument('--sft_models_dir', type=str, required=True, help='Directory containing the SFT models.')
    parser.add_argument('--output_file', type=str, required=True, help='Path of retrival JSON file.')
    args = parser.parse_args()

    main(args.model_lists, args.sft_models_dir, args.output_file)
