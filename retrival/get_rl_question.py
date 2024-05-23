import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
from trl import set_seed

def main(output_dir, k, seed):
    """
    Main function to load the dataset and extract the first k questions.
    """
    # Set the random seed for reproducibility
    set_seed(seed)

    dataset_name = "lvwerra/stack-exchange-paired"
    
    # Load the dataset
    train_dataset = load_dataset(dataset_name, split="train")
    train_dataset = train_dataset.select(range(k))
    original_columns = train_dataset.column_names

    print(f'Loaded dataset with {len(train_dataset)} entries.')

    q_data = [sample['question'] for sample in tqdm(train_dataset)]

    print(f'Collected {len(q_data)} questions.')

    # Save the questions to a JSON file
    output_path = f"{output_dir}/rl_questions.json"
    with open(output_path, "w") as json_file:
        json.dump(q_data, json_file)
        print(f'Dumped {len(q_data)} questions to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first k questions from the dataset.")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output JSON file.')
    parser.add_argument('--k', type=int, default=1000, help='Number of questions to extract.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    args = parser.parse_args()

    main(args.output_dir, args.k, args.seed)
