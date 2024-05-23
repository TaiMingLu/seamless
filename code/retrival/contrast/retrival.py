import argparse
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_data(data_dir, rl_questions_dir):
    """
    Load the required datasets and indices from the data directory.
    """
    rl_questions_path = f'{rl_questions_dir}/rl_questions.json'
    stact_dataset_path = f'{data_dir}/stack-exchange.json'
    stact_index_path = f'{data_dir}/stack_faiss.index'
    stact_index_indicis_path = f'{data_dir}/stack_faiss_index.json'

    with open(rl_questions_path, "r") as json_file:
        rl_questions = json.load(json_file)
        print('RL Questions Loaded')

    with open(stact_dataset_path, "r") as json_file:
        stact_dataset_dict = json.load(json_file)
        print('Dataset Loaded')

    with open(stact_index_indicis_path, "r") as json_file:
        stact_index = json.load(json_file)
        print('Dataset Indices Loaded')

    print('Number of RL questions:', len(rl_questions))
    print('Total database size:', len(stact_index))

    return rl_questions, stact_dataset_dict, stact_index, stact_index_path

def encode_and_search(encoder, index, stact_index, search_text, k, threshold=0.1):
    """
    Encode the search text and retrieve top-k results from the FAISS index,
    filtering by the distance threshold.
    """
    search_vector = encoder.encode(search_text)
    search_vector = search_vector / np.linalg.norm(search_vector)
    _vector = np.array([search_vector], dtype=np.float32)
    
    distances, indices = index.search(_vector, k + 10)
    distances = distances[0].tolist()
    indices = indices[0]
    
    filtered_indices = []
    filtered_distances = [] 
    
    for distance, idx in zip(distances, indices):
        if distance < 1 - threshold:
            filtered_indices.append(idx)
            filtered_distances.append(distance)
        if len(filtered_indices) == k:
            break
    
    if len(filtered_indices) < k:
        filtered_indices = indices[:k]
        filtered_distances = distances[:k]

    return filtered_indices, filtered_distances

def main(data_dir, rl_questions_dir, output_dir, model_name, k, threshold):
    """
    Main function to load data, encode texts, create FAISS index, and perform search.
    """
    rl_questions, stact_dataset_dict, stact_index, stact_index_path = load_data(data_dir, rl_questions_dir)
    
    encoder = SentenceTransformer(model_name)
    encoder = encoder.to('cuda')
    print('Encoder Loaded')

    index = faiss.read_index(stact_index_path)
    print('Index Loaded')
    print('Index total rows:', index.ntotal)

    data = []
    for q in tqdm(rl_questions):
        results = {
            'question': q,
            'gold_answer': [stact_dataset_dict[q], {'PM': {}, 'RM': {}}],
            'worse_answers': [],
            'num_hacking': {}
        }
        indices, distances = encode_and_search(encoder, index, stact_index, q, k, threshold)
        retrieved = []
        for i in range(len(indices)):
            retrieved.append([stact_dataset_dict[stact_index[indices[i]]], {'PM': {}, 'RM': {}, 'distance': distances[i]}])
        results['worse_answers'] = retrieved
        data.append(results)

    output_path = f"{output_dir}/SEAM_contrast.json"
    with open(output_path, "w") as json_file:
        json.dump(data, json_file)
        print('Dumped JSON')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data, encode texts, create FAISS index, and perform search.")
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory containing the JSON files.')
    parser.add_argument('--rl_questions_dir', type=str, required=True, help='Directory containing the RL questions JSON file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the results.')
    parser.add.argument('--model_name', type=str, default="sentence-transformers/multi-qa-distilbert-cos-v1", help='Model name to be used for encoding.')
    parser.add_argument('--k', type=int, default=30, help='Number of top results to retrieve.')
    parser.add.argument('--threshold', type=float, default=0.1, help='Distance threshold for filtering results.')
    args = parser.parse_args()

    main(args.data_dir, args.rl_questions_dir, args.output_dir, args.model_name, args.k, args.threshold)
