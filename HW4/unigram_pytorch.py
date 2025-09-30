"""Pytorch."""

import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt


FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def loss_fn(logp: float) -> float:
    """Compute loss to maximize probability."""
    return -logp


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct uniform initial s
        s0 = np.ones((V, 1))
        self.s = nn.Parameter(torch.tensor(s0.astype(float)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        logp = torch.nn.LogSoftmax(0)(self.s)

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ logp


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype(float))

    # Calculate empirical character frequencies (the optimal distribution)
    token_counts = np.sum(encodings, axis=1)
    total_tokens = np.sum(token_counts)
    optimal_probs = token_counts / total_tokens


    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations = 1000  # SET THIS
    learning_rate = 0.01    # SET THIS

    # Track losses during training
    losses = []
    min_losses = []  # Track minimum loss so far at each iteration

    # train model
    print("Training started...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for iteration in range(num_iterations):
        logp_pred = model(x)
        loss = loss_fn(logp_pred)
        
        # Store loss for visualization
        losses.append(loss.item())
        
        # Track minimum loss achieved so far
        if iteration == 0:
            min_losses.append(loss.item())
        else:
            min_losses.append(min(min_losses[-1], loss.item()))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Print progress
        if (iteration + 1) % 100 == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.6f}, Min Loss So Far: {min_losses[-1]:.6f}")

    print("Training complete!\n")

    with torch.no_grad():
        logp_final = torch.nn.LogSoftmax(0)(model.s)
        learned_probs = torch.exp(logp_final).numpy().flatten()

    plt.figure(figsize=(15, 6))
    
    x_pos = np.arange(len(vocabulary))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.bar(x_pos - width/2, optimal_probs, width, 
            label='Optimal (Empirical)', alpha=0.8, color='steelblue')
    plt.bar(x_pos + width/2, learned_probs, width, 
            label='Learned', alpha=0.8, color='coral')
    plt.xlabel('Token Index', fontsize=11)
    plt.ylabel('Probability', fontsize=11)
    plt.title('Token Probabilities: Optimal vs Learned', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    
    # Create labels for all tokens, handling special characters
    token_labels = []
    for i, token in enumerate(vocabulary):
        if token is None:
            token_labels.append('UNK')
        elif token == ' ':
            token_labels.append('SPACE')
        else:
            token_labels.append(token)
    
    # Show all tokens
    plt.xticks(x_pos, token_labels, rotation=45, fontsize=9, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()


    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Training Loss', linewidth=2, color='darkblue')
    plt.axhline(y=min(min_losses), color='red', linestyle='--', 
                label=f'Minimum Possible Loss', linewidth=2)
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('Loss (Negative Log-Likelihood)', fontsize=11)
    plt.title('Loss vs Iteration', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig('HW4/unigram_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to 'unigram_results.png'\n")


if __name__ == "__main__":
    gradient_descent_example()