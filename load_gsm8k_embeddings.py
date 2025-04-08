import pickle
import numpy as np
import torch

def load_embedding_dataset(file_path):
    '''Loads the embedding dataset from a JSON file.'''
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    # Convert the list representation of numpy arrays back to arrays
    formatted_data = [(q, s, c, e.cpu()) for q, s, c, e in data]
    return formatted_data


if __name__ == "__main__":
    data = load_embedding_dataset("embeddings.pkl")

