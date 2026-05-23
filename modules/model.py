import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from transformers import BertModel, BertConfig
import math

import torch.nn as nn

class FrequencyEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, num_frequencies, dropout=0.2):
        """
        Frequency-based embedding layer.

        Args:
            input_dim (int): Dimensionality of input data.
            embed_dim (int): Output embedding dimensionality.
            num_frequencies (int): Number of frequency components to use.
            dropout (float): Dropout rate for regularization.
        """
        super(FrequencyEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies

        # Learnable weights for frequency domain transformations
        self.freq_weights = nn.Parameter(torch.randn(num_frequencies, input_dim))
        
        # Fully connected layers for projection to embedding dimension
        self.fc = nn.Sequential(
            nn.Linear(num_frequencies, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass for frequency embedding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Apply Fourier Transform across the input dimension
        freq_components = torch.fft.rfft(x, dim=-1)  # Shape: (batch_size, seq_len, input_dim/2+1)

        # Take the magnitude (or other frequency features, if desired)
        freq_magnitudes = torch.abs(freq_components)  # Shape: (batch_size, seq_len, input_dim/2+1)
        
        # Select the top frequencies for embedding
        selected_freqs = freq_magnitudes[..., :self.num_frequencies]  # Shape: (batch_size, seq_len, num_frequencies)

        # Project into embedding space
        embedded = self.fc(selected_freqs)  # Shape: (batch_size, seq_len, embed_dim)
        return embedded


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_prob=0.2):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.ln2 = nn.LayerNorm(hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, num_classes)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.dropout(out)  # Apply Dropout after ReLU
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.dropout(out)  # Apply Dropout after ReLU
        out = self.output_layer(out)
        return out


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.fc(out)
        return out

class PositionalEncodingWithDecay(nn.Module):
    def __init__(self, max_seq_len, embedding_dim, decay_interval=9, decay_factor=0.9):
        super(PositionalEncodingWithDecay, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.decay_interval = decay_interval
        self.decay_factor = decay_factor
        
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        decay_steps = torch.floor(position / decay_interval).long()
        decay = decay_factor ** decay_steps.float()
        
        pe = pe * decay
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.size()
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        pe = self.pe[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + pe
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier_WithPositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, num_classes, 
                 embedding_dim=128, dropout=0.1, max_seq_len=900):
        """
        Initialize the Transformer classification model with positional encoding and Layer Normalization.
        
        :param input_dim: Input feature dimension (e.g., feature dimension of images or time series)
        :param num_heads: Number of heads in the multi-head attention mechanism
        :param num_layers: Number of Transformer Encoder layers
        :param hidden_dim: Hidden layer dimension in the Transformer
        :param num_classes: Number of classes for classification
        :param embedding_dim: Embedding dimension
        :param dropout: Dropout rate
        :param max_seq_len: Maximum sequence length for positional encoding
        """
        super(TransformerClassifier_WithPositionalEncoding, self).__init__()

        # Embedding layer: maps input_dim to embedding_dim
        self.embedding = nn.Linear(input_dim, embedding_dim)

        # LayerNorm layer
        self.layer_norm_input = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu'  # Use ReLU activation function
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        Forward propagation
        :param x: Input data, shape (batch_size, seq_len, input_dim)
        :return: Classification result
        """
        batch_size, seq_len = x.size()
        x = x.unsqueeze(-1)

        # Embedding layer
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim=128)

        # LayerNorm
        x = self.layer_norm_input(x)  # (batch_size, seq_len, embedding_dim=128)
        x = torch.relu(x)  # Activation function
        x = self.dropout(x)

        # Transformer encoding
        # Transformer expects input shape (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        x = self.transformer_encoder(x)  # (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embedding_dim)

        # Pooling: average pooling
        x = x.mean(dim=1)  # (batch_size, embedding_dim)

        # Classification
        output = self.fc(x)  # (batch_size, num_classes)

        return output


class BERT(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, num_classes, 
                 embedding_dim=128, dropout=0.1, max_seq_len=600):
        """
        Initialize the Transformer classification model with positional encoding and Layer Normalization.
        
        :param input_dim: Input feature dimension (e.g., feature dimension of images or time series)
        :param num_heads: Number of heads in the multi-head attention mechanism
        :param num_layers: Number of Transformer Encoder layers
        :param hidden_dim: Hidden layer dimension in the Transformer
        :param num_classes: Number of classes for classification
        :param embedding_dim: Embedding dimension
        :param dropout: Dropout rate
        :param max_seq_len: Maximum sequence length for positional encoding
        """
        super(BERT, self).__init__()

        # Embedding layer: maps input_dim to embedding_dim
        self.embedding = nn.Linear(input_dim, embedding_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncodingWithDecay(
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            decay_interval=9,
            decay_factor=0.9
        )

        # LayerNorm layer
        self.layer_norm_input = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

        # Define 8 identical Transformer Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='relu'
            )
            for _ in range(8)
        ])

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        Forward propagation
        :param x: Input data, shape (batch_size, seq_len, input_dim)
        :return: Classification result
        """
        batch_size, seq_len, input_dim = x.size()

        # Embedding layer
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim=128)

        # Add positional encoding
        x = self.positional_encoding(x)  # (batch_size, seq_len, embedding_dim=128)

        # LayerNorm
        x = self.layer_norm_input(x)  # (batch_size, seq_len, embedding_dim=128)
        x = torch.relu(x)  # Activation function
        x = self.dropout(x)

        # Transformer encoding
        # Transformer expects input shape (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        for encoder in self.encoder_layers:
            x = encoder(x)  # (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embedding_dim)

        # Pooling: average pooling
        x = x.mean(dim=1)  # (batch_size, embedding_dim)

        # Classification
        output = self.fc(x)  # (batch_size, num_classes)

        return output


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.output = nn.Sequential(
            nn.Linear(in_features=7200, out_features=4),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x
    
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConv1D_Attention(nn.Module):
    def __init__(self, input_width, num_classes, embedding_dim=256, num_heads=8, hidden_dim=512, num_layers=4, dropout=0.2):
        """
        Model combining deep Conv1D (short and long distance) and Attention for feature extraction and classification.
        
        Args:
        - input_width (int): Width of the input sequence (e.g., 900).
        - num_classes (int): Number of output classes.
        - embedding_dim (int): Dimension of embeddings for attention mechanism.
        - num_heads (int): Number of attention heads.
        - hidden_dim (int): Dimension of the feedforward layer in attention.
        - num_layers (int): Number of Transformer layers.
        - dropout (float): Dropout rate.
        """
        super(DeepConv1D_Attention, self).__init__()
        
        # Short-distance Conv1D with multiple layers
        self.short_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, stride=1, padding=0),  # Short filter
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Reduce width by half
        )
        
        # Long-distance Conv1D with multiple layers
        self.long_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=90, stride=1, padding=0),  # Long filter
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Reduce width by half
        )
        
        # Compute feature dimensions dynamically
        short_out_dim = (input_width // 9 // 2) * 64  # Output size from short Conv1D
        long_out_dim = ((input_width // 9 - 9 + 1) // 2) * 64  # Output size from long Conv1D
        original_out_dim = input_width  # Flattened original input
        
        feature_dim = 900 #55364  # Total concatenated features
        
        # Linear layer to map concatenated features to embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu'
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, width=900).
        
        Returns:
        - torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        batch_size, width = x.shape  # Shape: (batch_size, 900)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 900)
        
        # Short-distance Conv1D
        x_short = self.short_conv1d(x)  # Shape: (batch_size, 64, width/18)
        x_short = x_short.view(batch_size, -1)  # Flatten: (batch_size, 64 * width/18)
        
        # Long-distance Conv1D
        x_long = self.long_conv1d(x)  # Shape: (batch_size, 64, (width/9 - 9 + 1)/2)
        x_long = x_long.view(batch_size, -1)  # Flatten: (batch_size, 64 * (width/9 - 9 + 1)/2)
        
        # Flatten input for concatenation
        x_flat = x.view(batch_size, -1)  # Shape: (batch_size, 900)
        
        # Concatenate original, short, and long features
        x_concat = torch.cat([x_flat, x_short, x_long], dim=1)  # Shape: (batch_size, feature_dim)
        
        # Linear projection to embedding_dim
        x = self.embedding_layer(x_flat)  # Shape: (batch_size, embedding_dim)
        x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, seq_len=1, embedding_dim)
        
        # Attention
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, embedding_dim)
        x = self.attention(x)  # Shape: (seq_len=1, batch_size, embedding_dim)
        x = x.permute(1, 0, 2).squeeze(1)  # Back to (batch_size, embedding_dim)
        
        # Classification
        out = self.classifier(x)  # Shape: (batch_size, num_classes)
        out = torch.softmax(out, dim=1)  # Convert logits to probabilities
        return out


class Conv1D_Attention_Advanced(nn.Module):
    def __init__(self, input_width, num_classes, embedding_dim=256, num_heads=8, hidden_dim=512, num_layers=4, dropout=0.2):
        """
        Enhanced model combining Conv1D (short and long distance) and Attention for feature extraction and classification.
        
        Args:
        - input_width (int): Width of the input sequence (e.g., 900).
        - num_classes (int): Number of output classes.
        - embedding_dim (int): Dimension of embeddings for attention mechanism.
        - num_heads (int): Number of attention heads.
        - hidden_dim (int): Dimension of the feedforward layer in attention.
        - num_layers (int): Number of Transformer layers.
        - dropout (float): Dropout rate.
        """
        super(Conv1D_Attention_Advanced, self).__init__()
        
        # Short-distance 1D Convolutional Layers
        self.short_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Reduce width by half
            nn.Dropout(dropout)
        )
        
        # Long-distance 1D Convolutional Layers
        self.long_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Reduce width by half
            nn.Dropout(dropout)
        )
        
        # Linear layer to map concatenated features to embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.Linear(64 * (input_width // 2) + 64 * (input_width // 2), embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu'
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head with additional hidden layers
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, width=900).
        
        Returns:
        - torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        batch_size, width = x.shape  # Shape: (batch_size, 900)
        x = x.unsqueeze(1)  # Add a channel dimension for Conv1D (batch_size, 1, 900)
        
        # Short-distance 1D Convolution
        x_short = self.short_conv1d(x)  # Shape: (batch_size, 64, width/2)
        
        # Long-distance 1D Convolution
        x_long = self.long_conv1d(x)  # Shape: (batch_size, 64, width/2)
        
        # Flatten Conv1D outputs
        x_short = x_short.view(batch_size, -1)  # Shape: (batch_size, 64 * width/2)
        x_long = x_long.view(batch_size, -1)  # Shape: (batch_size, 64 * width/2)
        
        # Concatenate short and long 1D Conv features
        x_concat = torch.cat([x_short, x_long], dim=1)  # Shape: (batch_size, 64 * width/2 + 64 * width/2)
        
        # Linear projection to embedding_dim
        x = self.embedding_layer(x_concat)  # Shape: (batch_size, embedding_dim)
        x = x.unsqueeze(1)  # Shape: (batch_size, seq_len=1, embedding_dim)
        
        # Attention
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, embedding_dim)
        x = self.attention(x)  # Shape: (seq_len=1, batch_size, embedding_dim)
        x = x.permute(1, 0, 2).squeeze(1)  # Back to (batch_size, embedding_dim)
        
        # Classification
        out = self.classifier(x)  # Shape: (batch_size, num_classes)
        return out



def Sincos_position_encoding(seq_len, dim):
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))  # (dim//2,)
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
    return pe.unsqueeze(0)  # Shape: (1, seq_len, dim)

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Self-attention layer
        attn_output, _ = self.attn(x, x, x)
        x = attn_output + x  # Residual connection
        x = self.layer_norm(x)

        # Feed-forward network
        x = self.fc(x) + attn_output  # Residual connection
        return x


#---------------------------------------------------------------------Sincos---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttackDetectionModel_Sincos_hierarchical_attn(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2, num_heads=8):
        super(AttackDetectionModel_Sincos_hierarchical_attn, self).__init__()

        # Embedding layer for IDs
        self.id_embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutions for data
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=15, padding=7)

        # Attention Layer for ID features
        self.id_attn_layer = AttentionLayer(input_dim=embed_dim, hidden_dim=hidden_dim, num_heads=num_heads)

        # Attention Layer for Data features
        self.data_attn_layer = AttentionLayer(input_dim=hidden_dim * 2, hidden_dim=hidden_dim, num_heads=num_heads)

        # Sincos encoding (Sincos)
        self.id_pos_encoding = Sincos_position_encoding(100, embed_dim)
        self.data_pos_encoding = Sincos_position_encoding(100, hidden_dim * 2)

        # Fully connected layers after fusion
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        seq_len = x.size(1) // 9
        x = x.view(x.size(0), seq_len, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, seq_len)
        data = x[:, :, 1:]    # Shape: (batch_size, seq_len, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
        
        # Process ID with embedding and add positional encoding
        id_embed = self.id_embedding_layer(id_data)  # Shape: (batch_size, seq_len, embed_dim)
        id_embed = id_embed + self.id_pos_encoding.to(id_embed.device)  # Add positional encoding
        id_attn_out = self.id_attn_layer(id_embed)  # Shape: (batch_size, seq_len, embed_dim)
        id_pooled = torch.max(id_attn_out, dim=1).values  # Shape: (batch_size, embed_dim)

        # Process data with convolution and add positional encoding
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, seq_len)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, seq_len)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim*2, seq_len)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, seq_len, hidden_dim*2)
        data_feat = data_feat + self.data_pos_encoding.to(data_feat.device)  # Add positional encoding
        data_attn_out = self.data_attn_layer(data_feat)  # Shape: (batch_size, seq_len, hidden_dim*2)
        data_pooled = torch.max(data_attn_out, dim=1).values  # Shape: (batch_size, hidden_dim*2)

        # Concatenate ID and data features
        combined_feat = torch.cat((id_pooled, data_pooled), dim=-1)  # Shape: (batch_size, embed_dim + hidden_dim*2)

        # Classification
        out = self.fc(combined_feat)  # Shape: (batch_size, num_classes)
        
        return out


class AttackDetectionModel_Sincos(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2, num_heads=4):
        super(AttackDetectionModel_Sincos, self).__init__()

        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutions for data
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)

        # Attention Layer
        self.attn_layer = AttentionLayer(input_dim=embed_dim + hidden_dim * 2, hidden_dim=hidden_dim, num_heads=num_heads)
        
        # Sincos encoding (Sincos)
        self.pos_encoding = Sincos_position_encoding(100, embed_dim + hidden_dim * 2)  # Assume sequence length is 100
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim*2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim*2)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Add positional encoding (Sincos)
        combined_feat = combined_feat + self.pos_encoding.to(combined_feat.device)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Apply Attention
        attention_out = self.attn_layer(combined_feat)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, embed_dim + hidden_dim*2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out

class AttackDetectionModel_Sincos_Fre_EMB(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2, num_heads=4):
        super(AttackDetectionModel_Sincos_Fre_EMB, self).__init__()

        # Embedding layer for IDs
        self.embedding_layer1 = FrequencyEmbedding(
            input_dim=input_width,
            embed_dim=embed_dim // 2,
            num_frequencies=1,
            dropout=dropout
        )
        self.embedding_layer2 = nn.Sequential(
            nn.Linear(input_width, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutions for data
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)

        # Attention Layer
        self.attn_layer = AttentionLayer(input_dim=embed_dim + hidden_dim * 2, hidden_dim=hidden_dim, num_heads=num_heads)
        
        # Sincos encoding (Sincos)
        self.pos_encoding = Sincos_position_encoding(100, embed_dim + hidden_dim * 2)  # Assume sequence length is 100
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed1 = self.embedding_layer1(id_data)  # Shape: (batch_size, 100, embed_dim)
        id_embed2 = self.embedding_layer2(id_data)
        id_embed = torch.cat((id_embed1, id_embed2), dim=-1)

        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim*2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim*2)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Add positional encoding (ROPE)
        combined_feat = combined_feat + self.pos_encoding.to(combined_feat.device)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Apply Attention
        attention_out = self.attn_layer(combined_feat)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, embed_dim + hidden_dim*2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out

class AttackDetectionModel_Sincos_Fre(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2, num_heads=4):
        super(AttackDetectionModel_Sincos_Fre, self).__init__()

        # Embedding layer for IDs
        self.embedding_layer = FrequencyEmbedding(
            input_dim=input_width,
            embed_dim=embed_dim,
            num_frequencies=1,
            dropout=dropout
        )
        
        # Multiple 1D convolutions for data
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)

        # Attention Layer
        self.attn_layer = AttentionLayer(input_dim=embed_dim + hidden_dim * 2, hidden_dim=hidden_dim, num_heads=num_heads)
        
        # Sincos encoding (Sincos)
        self.pos_encoding = Sincos_position_encoding(100, embed_dim + hidden_dim * 2)  # Assume sequence length is 100
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)

        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim*2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim*2)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Add positional encoding (ROPE)
        combined_feat = combined_feat + self.pos_encoding.to(combined_feat.device)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Apply Attention
        attention_out = self.attn_layer(combined_feat)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, embed_dim + hidden_dim*2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out

class AttackDetectionModel_no_pos(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2, num_heads=4):
        super(AttackDetectionModel_no_pos, self).__init__()

        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutions for data
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)

        # Attention Layer
        self.attn_layer = AttentionLayer(input_dim=embed_dim + hidden_dim * 2, hidden_dim=hidden_dim, num_heads=num_heads)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim*2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim*2)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Apply Attention
        attention_out = self.attn_layer(combined_feat)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, embed_dim + hidden_dim*2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out

class AttackDetectionModel_no_attn(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2, num_heads=4):
        super(AttackDetectionModel_no_attn, self).__init__()

        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutions for data
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)

        # Attention Layer
        self.attn_layer = AttentionLayer(input_dim=embed_dim + hidden_dim * 2, hidden_dim=hidden_dim, num_heads=num_heads)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim*2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim*2)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(combined_feat, dim=1).values  # Shape: (batch_size, embed_dim + hidden_dim*2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out

class AttackDetectionModel_no_conv_pos(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2, num_heads=4):
        super(AttackDetectionModel_no_conv_pos, self).__init__()

        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention Layer
        self.attn_layer = AttentionLayer(input_dim=embed_dim + data_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + data_dim, 64),  # Concatenate ID and data features
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Use raw data directly (no convolution)
        data_feat = data  # Use raw data directly
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + data_dim)
        
        # Apply Attention
        attention_out = self.attn_layer(combined_feat)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, embed_dim + hidden_dim*2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out


class AttackDetectionModel_no_embedding_pos(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2, num_heads=4):
        super(AttackDetectionModel_no_embedding_pos, self).__init__()

        # Ensure that hidden_dim and embed_dim add up to 384 for concatenation
        self.embed_dim = embed_dim  # 256 as per your configuration
        self.hidden_dim = hidden_dim  # 64 as per your configuration
        
        # Multiple 1D convolutions for data
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)

        # Attention Layer (Input dimension should be embed_dim + hidden_dim*2)
        self.attn_layer = AttentionLayer(input_dim=hidden_dim * 4, hidden_dim=hidden_dim, num_heads=num_heads)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),  # Concatenate ID and data features
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # We no longer use embedding for ID, use raw ID data directly
        id_embed = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Expand id_embed to match the data's feature dimensions, i.e., (batch_size, 100, hidden_dim*2)
        id_embed_expanded = id_embed.expand(-1, -1, self.hidden_dim * 2)  # Shape: (batch_size, 100, hidden_dim*2)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim*2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim*2)

        # Concatenate expanded ID and data features
        combined_feat = torch.cat((id_embed_expanded, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)

        # Apply Attention
        attention_out = self.attn_layer(combined_feat)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, embed_dim + hidden_dim*2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out


# ---------------------------------------------------------ROPE ablation---------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary positional embeddings to query and key tensors.
    """
    # Reshape and convert to complex domain
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # Apply rotation and convert back to real domain
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    Precompute rotary positional embeddings for given dimensions and sequence length.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, seq_len, dropout=0.1):
        """
        Multi-head attention layer with rotary positional embeddings.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_model, d_model)

        self.freqs_cis = precompute_freqs_cis(d_model, seq_len).to(torch.device("cuda:2"))

    def forward(self, query, key, value):
        """
        Apply multi-head attention with rotary positional embeddings.
        """
        batch_size, seq_len, _ = query.size()

        # Linear projections
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        # Apply rotary embeddings
        query, key = apply_rotary_emb(query, key, self.freqs_cis)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)

        # Concatenate heads and apply final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output_linear(attn_output)
        return output


class AttackDetectionModel(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2, num_heads=4):
        super(AttackDetectionModel, self).__init__()

        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multiple 1D convolutions for data
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)

        # Multi-head attention layer with RoPE
        self.attn_layer = MultiHeadedAttention(num_heads=num_heads, d_model=embed_dim + hidden_dim * 2, seq_len=100, dropout=dropout)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)

        # Split ID and data
        id_data = x[:, :, 0].unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        data = x[:, :, 1:]  # Shape: (batch_size, 100, 8)

        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)

        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim*2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim*2)

        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)

        # Apply Attention
        attention_out = self.attn_layer(combined_feat, combined_feat, combined_feat)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)

        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, embed_dim + hidden_dim*2)

        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)

        return out


class AttackDetectionModel_no_conv(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2, num_heads=4):
        super(AttackDetectionModel_no_conv, self).__init__()

        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention Layer
        self.attn_layer = AttentionLayer(input_dim=embed_dim + data_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        
        # ROPE encoding
        self.pos_encoding = rope_position_encoding(100, embed_dim + hidden_dim * 2)  # Assume sequence length is 100

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + data_dim, 64),  # Concatenate ID and data features
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Use raw data directly (no convolution)
        data_feat = data  # Use raw data directly
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + data_dim)
        
        # Add positional encoding (ROPE)
        combined_feat = combined_feat + self.pos_encoding.to(combined_feat.device)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)

        # Apply Attention
        attention_out = self.attn_layer(combined_feat)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, embed_dim + hidden_dim*2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out


class AttackDetectionModel_no_embedding(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2, num_heads=4):
        super(AttackDetectionModel_no_embedding, self).__init__()

        # Ensure that hidden_dim and embed_dim add up to 384 for concatenation
        self.embed_dim = embed_dim  # 256 as per your configuration
        self.hidden_dim = hidden_dim  # 64 as per your configuration
        
        # Multiple 1D convolutions for data
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)

        # Attention Layer (Input dimension should be embed_dim + hidden_dim*2)
        self.attn_layer = AttentionLayer(input_dim=hidden_dim * 4, hidden_dim=hidden_dim, num_heads=num_heads)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),  # Concatenate ID and data features
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # We no longer use embedding for ID, use raw ID data directly
        id_embed = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Expand id_embed to match the data's feature dimensions, i.e., (batch_size, 100, hidden_dim*2)
        id_embed_expanded = id_embed.expand(-1, -1, self.hidden_dim * 2)  # Shape: (batch_size, 100, hidden_dim*2)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim*2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim*2)

        # Concatenate expanded ID and data features
        combined_feat = torch.cat((id_embed_expanded, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)

        # Add positional encoding (ROPE)
        combined_feat = combined_feat + self.pos_encoding.to(combined_feat.device)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)

        # Apply Attention
        attention_out = self.attn_layer(combined_feat)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, embed_dim + hidden_dim*2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out


class AttackDetectionModel_LSTM(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2):
        super(AttackDetectionModel_LSTM, self).__init__()
        # Embedding layer for IDs
        self.embedding_layer = FrequencyEmbedding(
            input_dim=input_width,
            embed_dim=embed_dim,
            num_frequencies=1,
            dropout=dropout
        )
        
        # 1D convolution for data
        self.data_conv = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=embed_dim + hidden_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim)
        
        # Sequence modeling with LSTM
        lstm_out, _ = self.lstm(combined_feat)  # Shape: (batch_size, 100, hidden_dim * 2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(lstm_out, dim=1).values  # Shape: (batch_size, hidden_dim * 2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out

class OptimizedAttackDetectionModel(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=128, num_classes=4, dropout=0.3, num_heads=4):
        super(OptimizedAttackDetectionModel, self).__init__()
        
        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutional layers for feature extraction
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=embed_dim + hidden_dim * 2, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        
        # Multi-head Attention for temporal dependencies
        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads, batch_first=True)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim * 2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim * 2)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim * 2)
        
        # Sequence modeling with LSTM
        lstm_out, _ = self.lstm(combined_feat)  # Shape: (batch_size, 100, hidden_dim * 2)
        
        # Attention mechanism
        attention_out, _ = self.attention_layer(lstm_out, lstm_out, lstm_out)  # Shape: (batch_size, 100, hidden_dim * 2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, hidden_dim * 2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out

from mamba_ssm import Mamba
class MambaCAN(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=128, num_classes=4, dropout=0.3, num_heads=4):
        super(MambaCAN, self).__init__()
        
        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutional layers for feature extraction
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        
        # Mamba
        self.MambaLayer = Mamba(d_model=embed_dim + hidden_dim * 2, d_state=16, d_conv=4, expand=2,)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim * 2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim * 2)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim * 2)
        
        # Attention mechanism
        attention_out = self.MambaLayer(combined_feat)  # Shape: (batch_size, 100, hidden_dim * 2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, hidden_dim * 2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out

class MambaCAN_noconv(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=128, num_classes=4, dropout=0.3, num_heads=4):
        super(MambaCAN_noconv, self).__init__()
        
        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutional layers for feature extraction
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        
        # Mamba
        self.MambaLayer = Mamba(d_model=264, d_state=16, d_conv=4, expand=2,)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(264, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim * 2)
        
        # Attention mechanism
        attention_out = self.MambaLayer(combined_feat)  # Shape: (batch_size, 100, hidden_dim * 2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, hidden_dim * 2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out


class MambaCAN_noid(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=128, num_classes=4, dropout=0.3, num_heads=4):
        super(MambaCAN_noid, self).__init__()
        
        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutional layers for feature extraction
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        
        # Mamba
        self.MambaLayer = Mamba(d_model=257, d_state=16, d_conv=4, expand=2,)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(257, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)

        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim * 2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim * 2)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_data, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim * 2)
        
        # Attention mechanism
        attention_out = self.MambaLayer(combined_feat)  # Shape: (batch_size, 100, hidden_dim * 2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, hidden_dim * 2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out


class MambaCAN_Only(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=128, num_classes=4, dropout=0.3, num_heads=4):
        super(MambaCAN_Only, self).__init__()
        
        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutional layers for feature extraction
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        
        # Mamba
        self.MambaLayer = Mamba(d_model=9, d_state=16, d_conv=4, expand=2,)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Attention mechanism
        attention_out = self.MambaLayer(x)  # Shape: (batch_size, 100, hidden_dim * 2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, hidden_dim * 2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out

class MambaCAN_2Direction_1conv(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=128, num_classes=4, dropout=0.3, num_heads=4):
        super(MambaCAN_2Direction_1conv, self).__init__()
        
        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutional layers for feature extraction
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        
        # Mamba
        self.MambaLayer_forward = Mamba(d_model=embed_dim + hidden_dim * 2, d_state=8, d_conv=4, expand=2,)
        self.MambaLayer_backward = Mamba(d_model=embed_dim + hidden_dim * 2, d_state=8, d_conv=4, expand=2,)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim * 2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim * 2)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim * 2)

        forward_out = self.MambaLayer_forward(combined_feat)  # Forward Mamba: (batch size, 100, hidden_dim * 2)
        backward_out = self.MambaLayer_backward(torch.flip(combined_feat, dims=[1]))  # Backward Mamba: (batch size, 100, hidden_dim * 2)
        backward_out = torch.flip(backward_out, dims=[1])  # Flip back to original sequence order
        # Mamba mechanism
        Mamba_out = forward_out + backward_out
        
        # Max pooling over sequence
        pooled_feat = torch.max(Mamba_out, dim=1).values  # Shape: (batch_size, hidden_dim * 2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out


class MambaCAN_2Direction(nn.Module):

    
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=128, num_classes=4, dropout=0.3, num_heads=4):
        super(MambaCAN_2Direction, self).__init__()
        
        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutional layers for feature extraction
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        
        # Mamba
        self.MambaLayer_forward = Mamba(d_model=embed_dim + hidden_dim * 2, d_state=16, d_conv=4, expand=2,)
        self.MambaLayer_backward = Mamba(d_model=embed_dim + hidden_dim * 2, d_state=8, d_conv=2, expand=2,)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim * 2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim * 2)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim * 2)

        forward_out = self.MambaLayer_forward(combined_feat)  # Forward Mamba: (batch size, 100, hidden_dim * 2)
        backward_out = self.MambaLayer_backward(torch.flip(combined_feat, dims=[1]))  # Backward Mamba: (batch size, 100, hidden_dim * 2)
        backward_out = torch.flip(backward_out, dims=[1])  # Flip back to original sequence order
        # Mamba mechanism
        Mamba_out = self.alpha * forward_out + self.beta * backward_out
        
        # Max pooling over sequence
        pooled_feat = torch.max(Mamba_out, dim=1).values  # Shape: (batch_size, hidden_dim * 2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out

class MambaCAN_2Direction_Fre(nn.Module):

    
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=128, num_classes=4, dropout=0.3, num_heads=4):
        super(MambaCAN_2Direction_Fre, self).__init__()
        
        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            FrequencyEmbedding(
            input_dim=input_width,
            embed_dim=embed_dim,
            num_frequencies=1,
            dropout=dropout
            ),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Multiple 1D convolutional layers for feature extraction
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        
        # Mamba
        self.MambaLayer_forward = Mamba(d_model=embed_dim + hidden_dim * 2, d_state=16, d_conv=4, expand=2,)
        self.MambaLayer_backward = Mamba(d_model=embed_dim + hidden_dim * 2, d_state=8, d_conv=2, expand=2,)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim * 2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim * 2)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim * 2)

        forward_out = self.MambaLayer_forward(combined_feat)  # Forward Mamba: (batch size, 100, hidden_dim * 2)
        backward_out = self.MambaLayer_backward(torch.flip(combined_feat, dims=[1]))  # Backward Mamba: (batch size, 100, hidden_dim * 2)
        backward_out = torch.flip(backward_out, dims=[1])  # Flip back to original sequence order
        # Mamba mechanism
        Mamba_out = forward_out + backward_out
        
        # Max pooling over sequence
        pooled_feat = torch.max(Mamba_out, dim=1).values  # Shape: (batch_size, hidden_dim * 2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out