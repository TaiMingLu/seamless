import argparse
from tqdm import tqdm
from datasets import load_dataset
import json

def get_stack_data(save_dir):
    print('Start Loading Dataset')
    train_dataset = load_dataset("lvwerra/stack-exchange-paired", split="train", streaming=True)
    test_dataset = load_dataset("lvwerra/stack-exchange-paired", split="test", streaming=True)
    data = {}

    for sample in tqdm(train_dataset, desc="Loading train dataset"):
        question = sample['question']
        response = sample['response_j']
        data[question] = response

    for sample in tqdm(test_dataset, desc="Loading test dataset"):
        question = sample['question']
        response = sample['response_j']
        data[question] = response

    save_path = f"{save_dir}/stack-exchange.json"
    with open(save_path, "w") as json_file:
        json.dump(data, json_file)

    print('Number of Unique Questions:', len(data.keys()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Stack Exchange dataset and save to a JSON file.")
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the JSON file.')
    args = parser.parse_args()

    get_stack_data(args.save_dir)
