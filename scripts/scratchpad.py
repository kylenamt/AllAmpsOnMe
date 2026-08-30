#load the embeddings
# calculate the magnitude of the embeddings
# calculate the mean of the magnitudes
# calculate the median of the magnitudes

import torch
import ipdb
from pathlib import Path

# ipdb.set_trace()

enrolled_embeddings_path = Path('./results/emulate/delta_a2_tanh_r32_512/enroll/enrolled_embeddings.pt')  # Path to the embeddings file
embeddings_path = Path('./results/emulate/delta_a2_tanh_r32_512/embedding.pt')  # Path to the embeddings file
enrolled_embeddings = torch.load(enrolled_embeddings_path)['embedding']  # Load the enrolled embeddings from a file
embeddings = torch.load(embeddings_path)['embedding']  # Load the embeddings from a file


trained_magnitudes = [torch.norm(emb) for emb in embeddings]  # Calculate the magnitude of each embedding
mean_trained_magnitude = torch.mean(torch.stack(trained_magnitudes))  # Calculate the mean of the magnitudes
median_trained_magnitude = torch.median(torch.stack(trained_magnitudes))  # Calculate the median of the magnitudes
print(f'Mean Trained Magnitude: {mean_trained_magnitude.item()}')  # Print the mean magnitude
print(f'Median Trained Magnitude: {median_trained_magnitude.item()}')

enrolled_magnitudes = [torch.norm(emb) for emb in enrolled_embeddings]  # Calculate the magnitude of each enrolled embedding
mean_magnitude = torch.mean(torch.stack(enrolled_magnitudes))  # Calculate the mean of the magnitudes
median_magnitude = torch.median(torch.stack(enrolled_magnitudes))  # Calculate the median of the magnitudes
print(f'Mean Magnitude: {mean_magnitude.item()}')  # Print the mean magnitude
print(f'Median Magnitude: {median_magnitude.item()}')  # Print the median magnitude

