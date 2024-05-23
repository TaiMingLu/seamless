import argparse
import json
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np
import faiss

def encode_texts_in_batches(model, texts, pool, batch_size=5000):
    all_vectors = []
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)
    for batch_num in tqdm(range(total_batches), desc="Encoding"):
        start_index = batch_num * batch_size
        end_index = start_index + batch_size
        batch_texts = texts[start_index:end_index]
        vectors = model.encode_multi_process(batch_texts, pool)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        all_vectors.append(vectors)
    all_vectors = np.concatenate(all_vectors, axis=0)
    return all_vectors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode text and create FAISS index.")
    parser.add_argument('--model_name', type=str, default="sentence-transformers/multi-qa-distilbert-cos-v1", help='Model name to be used for encoding.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the FAISS index and JSON files.')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for encoding texts.')
    args = parser.parse_args()

    encoder = SentenceTransformer(args.model_name)
    pool = encoder.start_multi_process_pool()
    print('Encoder Loaded')

    with open(f"{args.save_dir}/stack-exchange.json", "r") as json_file:
        stact_dataset_dict = json.load(json_file)
        stact_dataset = list(stact_dataset_dict.keys())
        del stact_dataset_dict
    print('Dataset Loaded')

    print('Total Dataset size:', len(stact_dataset))

    vectors = encode_texts_in_batches(encoder, stact_dataset, pool, args.batch_size)

    d = 768  # Dimension of vectors
    # Use IndexFlatIP for cosine similarity
    index = faiss.IndexFlatIP(d)
    index.add(vectors)  # Vectors are already normalized

    faiss.write_index(index, f"{args.save_dir}/stack_faiss.index")
    print('Total index:', index.ntotal)

    with open(f"{args.save_dir}/stack_faiss_index.json", "w") as json_file:
        json.dump(stact_dataset, json_file)
        print('Dumped JSON')

    # Example query
    search_text = '<p>I know that here at SE I cant ask recommendation of products and so. I have a printer that I wanna buy and I want to know if it is good. Where should I post this question in order to get some smart people to take a look?</p>'
    print(f'Example Query: '{search_text})
    search_vector = encoder.encode(search_text)
    # Normalize the query vector
    search_vector = search_vector / np.linalg.norm(search_vector)
    _vector = np.array([search_vector], dtype=np.float32)

    # Search the index
    distances, ann = index.search(_vector, k=30)
    print(ann[0])
    for i in ann[0]:
        print(stact_dataset[i])
