from datasets import load_dataset
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from sentence_transformers import SentenceTransformer
from promptcot import generate_concepts

# Check for MPS availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def embed_concepts_batched(questions, solutions, embedding_model, batch_size=4):
    '''Batch embedding calculation for a list of (question, solution) pairs'''
    all_concepts = []

    # Generate concepts for each question-solution pair
    for question, solution in zip(questions, solutions):
        concepts = generate_concepts(question, solution, 5, "gpt-4-turbo")
        all_concepts.extend(concepts)
        print("DONE")

    # Batch the concepts for efficient embedding calculation
    batched_embeddings = []
    for i in range(0, len(all_concepts), batch_size):
        batch = all_concepts[i:i + batch_size]
        embeddings = embedding_model.encode(batch, convert_to_tensor=True, device=device)
        batched_embeddings.append(embeddings)

    # Concatenate all batches and compute the mean per concept group
    all_embeddings = torch.cat(batched_embeddings, dim=0)

    # Reshape and calculate mean embeddings for each question-solution pair
    mean_embeddings = []
    for i in range(0, len(all_embeddings), 5):  # Each question-solution has 5 concepts
        mean_embedding = torch.mean(all_embeddings[i:i+5], dim=0)
        mean_embeddings.append(mean_embedding.cpu().numpy())

    return np.array(mean_embeddings)

if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    embedding_model.to(device)  # Move model to MPS

    # Load the MATH-500 dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    train_data = dataset['test']

    N_SAMPLES = 100
    batch_size = 16  # You can adjust this based on available MPS memory

    # Prepare questions and solutions
    questions = [train_data[i]['problem'] for i in range(N_SAMPLES)]
    solutions = [train_data[i]['solution'] for i in range(N_SAMPLES)]

    # Get batched embeddings
    embedding_list = embed_concepts_batched(questions, solutions, embedding_model, batch_size)

    print(f"Extracted embeddings shape: {embedding_list.shape}")

    # Perform t-SNE on the CPU as sklearn does not support GPU
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    reduced_embeddings = tsne.fit_transform(embedding_list)

    # Plotting the reduced embeddings in 3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]
    z = reduced_embeddings[:, 2]

    # Scatter plot with labels (if available)
    ax.scatter(x, y, z, c='blue', marker='o')

    ax.set_title('3D t-SNE Plot of Embeddings')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')

    plt.show()
