import argparse
import json
import os
from openai import OpenAI
from tqdm import tqdm

def set_api_key(api_key):
    """
    Set the OpenAI API key as an environment variable.
    """
    os.environ['OPENAI_API_KEY'] = api_key

def load_data(file_path):
    """
    Load data from a JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_prompt(question, answer):
    """
    Create a prompt for generating a lower-quality response.
    """
    return f"""
Using the question and its correct answer provided below, generate an answer that is of
lower quality. The response should include one or more of the following characteristics: factual
inaccuracies, misunderstandings of the core question, irrelevant information, or grammatical errors.
The answers should vary in their mistakes to cover a range of common errors seen in similar topics.
[Question]
{question}
[Answer]
{answer}
[The Start of Assistant’s Answer]
{answer}
[The End of Assistant’s Answer]
"""

def get_completion(client, question, answer, model_name, k=1):
    """
    Generate lower-quality answers using OpenAI's API.
    """
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You generate responses to questions with lower quality. Your generated responses are long."},
            {"role": "user", "content": get_prompt(question, answer)}
        ],
        n=k,
        temperature=1
    )
    return [completion.choices[i].message.content for i in range(k)]

def main(api_key, rl_questions_dir, output_dir, k, model_name, data_dir):
    """
    Main function to load data, generate responses, and save the results.
    """

    stact_dataset_path = f'{data_dir}/stack-exchange.json'
    with open(stact_dataset_path, "r") as json_file:
        stact_dataset_dict = json.load(json_file)
        print('Dataset Loaded')

    rl_questions = load_data(f'{rl_questions_dir}/rl_questions.json')
    data = []
    for q in tqdm(rl_questions):
        results = {
            'question': q,
            'gold_answer': [stact_dataset_dict[q], {'PM': {}, 'RM': {}}],
            'worse_answers': [],
            'num_hacking': {}
        }
        data.append(results)


    set_api_key(api_key)
    client = OpenAI()

    print(f"Loaded {len(data)} entries from {rl_questions_dir}.")

    for i in tqdm(range(len(data))):
        question = data[i]['question']
        answer = data[i]['gold_answer'][0]
        results = get_completion(client, question, answer, model_name, k)
        worse_answers = [[results[j], {}] for j in range(k)]
        data[i]['worse_answers'] = worse_answers

    output_file = os.path.join(output_dir, 'SEAM_gpt.json')
    with open(output_file, 'w') as file:
        json.dump(data, file)
    print(f'Dumped {len(data)} entries with generated responses to {output_file}.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate lower-quality responses for questions using OpenAI API.")
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key.')
    parser.add_argument('--rl_questions_dir', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory containing the JSON files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output JSON file.')
    parser.add_argument('--k', type=int, default=30, help='Number of responses to generate per question.')
    parser.add.argument('--openai_model_name', type=str, default='gpt-4', help='OpenAI model name to be used for generating responses.')
    args = parser.parse_args()

    main(args.api_key, args.rl_questions_dir, args.output_dir, args.k, args.openai_model_name, args.data_dir)
