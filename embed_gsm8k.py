from datasets import load_dataset
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from sentence_transformers import SentenceTransformer
import pickle

from promptcot import generate_concepts

def create_embedding_dataset(questions, solutions, embedding_model, output_file):
    '''Creates a dataset of (question, solution, concept embeddings) tuples and saves to a file.'''

    data = []
    for question, solution in zip(questions, solutions):
        concepts = generate_concepts(question, solution, 5, "gpt-4-turbo")
        if len(concepts) != 5:
            print("Did not generate exactly 5 concepts")
            continue
        embeddings = embedding_model.encode(concepts, convert_to_tensor=True, device=device)
        data.append((question, solution, concepts, embeddings))

    with open(output_file, "wb") as file:
        pickle.dump(data, file)

    print(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    load_dotenv()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    embedding_model.to(device)  # Move model to MPS

    # Load the MATH-500 dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    train_data = dataset['test']

    N_SAMPLES = 2

    # Prepare questions and solutions
    questions = [train_data[i]['problem'] for i in range(N_SAMPLES)]
    solutions = [train_data[i]['solution'] for i in range(N_SAMPLES)]

    create_embedding_dataset(questions, solutions, embedding_model,
                             "./embeddings.pkl")
