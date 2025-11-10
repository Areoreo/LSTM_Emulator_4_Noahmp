"""
LSTM model for parameter prediction from simulation results
"""

import torch
import torch.nn as nn

class LSTMParameterPredictor(nn.Module):
    """
    LSTM-based model to predict parameters from time series data

    Architecture:
    - LSTM layers to process time series
    - Fully connected layers to predict parameters
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        Args:
            input_dim: Number of input variables (e.g., 4 for SOIL_M, LH, HFX, doy)
            hidden_dim: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_dim: Number of parameters to predict
            dropout: Dropout rate
        """
        super(LSTMParameterPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # LSTM forward pass
        # lstm_out: (batch_size, seq_len, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        # h_n[-1]: (batch_size, hidden_dim)
        last_hidden = h_n[-1]

        # Fully connected layers
        output = self.fc_layers(last_hidden)

        return output


class BiLSTMParameterPredictor(nn.Module):
    """
    Bidirectional LSTM-based model for parameter prediction
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        Args:
            input_dim: Number of input variables
            hidden_dim: Number of hidden units in LSTM (per direction)
            num_layers: Number of LSTM layers
            output_dim: Number of parameters to predict
            dropout: Dropout rate
        """
        super(BiLSTMParameterPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Fully connected layers (input is 2*hidden_dim due to bidirectional)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Concatenate last hidden states from both directions
        # h_n shape: (num_layers * 2, batch_size, hidden_dim)
        hidden_fwd = h_n[-2]  # Forward direction
        hidden_bwd = h_n[-1]  # Backward direction
        last_hidden = torch.cat([hidden_fwd, hidden_bwd], dim=1)

        # Fully connected layers
        output = self.fc_layers(last_hidden)

        return output


class AttentionLSTMParameterPredictor(nn.Module):
    """
    LSTM-based model with attention mechanism for parameter prediction

    The attention mechanism learns to focus on the most relevant timesteps
    in the sequence, potentially improving prediction accuracy.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        Args:
            input_dim: Number of input variables
            hidden_dim: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_dim: Number of parameters to predict
            dropout: Dropout rate
        """
        super(AttentionLSTMParameterPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        # Learns to compute attention scores for each timestep
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )

    def forward(self, x):
        """
        Forward pass with attention mechanism

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # LSTM forward pass
        # lstm_out: (batch_size, seq_len, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Compute attention scores for each timestep
        # attention_scores: (batch_size, seq_len, 1)
        attention_scores = self.attention(lstm_out)

        # Apply softmax to get attention weights
        # attention_weights: (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Compute context vector as weighted sum of LSTM outputs
        # context: (batch_size, hidden_dim)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Fully connected layers
        output = self.fc_layers(context)

        return output
