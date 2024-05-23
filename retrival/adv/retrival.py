import argparse
import json
import random
from tqdm import tqdm
from textattack.augmentation import Augmenter
from textattack.transformations import (
    CompositeTransformation, WordSwapChangeLocation,
    WordSwapRandomCharacterDeletion, WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution, WordSwapChangeName,
    WordInsertionMaskedLM, BackTranslation
)
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification

def setup_augmenter():
    """
    Setup a composite augmenter with various transformations and constraints.
    """
    # Composite transformation with various transformations
    transformation = CompositeTransformation([
        BackTranslation(), WordSwapChangeLocation(),
        WordSwapRandomCharacterDeletion(), WordSwapRandomCharacterInsertion(),
        WordSwapRandomCharacterSubstitution(), WordSwapChangeName(),
        WordInsertionMaskedLM()
    ])
    constraints = [RepeatModification(), StopwordModification()]
    augmenter = Augmenter(
        transformation=transformation,
        pct_words_to_swap=0.3,
        transformations_per_example=100,
        high_yield=True,
        constraints=constraints
    )
    return augmenter

def get_results(s, augmenter):
    """
    Apply the augmenter to generate adversarial examples.
    """
    results = augmenter.augment(s)
    random.shuffle(results)
    return results

def main(rl_questions_dir, data_dir, output_dir):
    """
    Main function to load data, generate adversarial examples, and save the results.
    """
    augmenter = setup_augmenter()

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


    final_results = []

    output_file = os.path.join(output_dir, 'SEAM_adv.json')
    for i in tqdm(range(len(data))):
        gold = data[i]['gold_answer'][0]
        adv = get_results(gold, augmenter)
        final_results.append({'gold': gold, 'adv': adv})

        with open(output_file, 'w') as file:
            json.dump(final_results, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate adversarial examples for SEAM data.")
    parser.add_argument('--rl_questions_dir', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory containing the JSON files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output JSON file.')
    args = parser.parse_args()

    main(args.rl_questions_dir, args.data_dir, args.output_dir)
