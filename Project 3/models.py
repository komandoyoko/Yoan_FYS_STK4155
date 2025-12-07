# models.py

from __future__ import annotations

from typing import Literal, Tuple, Optional

import torch
from torch import nn

from config import cfg


RNNType = Literal["rnn", "lstm", "gru"]


def _make_rnn_module(
    rnn_type: RNNType,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float = 0.0,
    bidirectional: bool = False,
    batch_first: bool = True,
) -> nn.Module:
    """
    Factory helper to create RNN / LSTM / GRU modules with common args.
    """
    rnn_type = rnn_type.lower()
    common_kwargs = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        dropout=dropout if num_layers > 1 else 0.0,  # PyTorch ignores dropout if num_layers=1
        bidirectional=bidirectional,
    )

    if rnn_type == "rnn":
        return nn.RNN(**common_kwargs)
    elif rnn_type == "lstm":
        return nn.LSTM(**common_kwargs)
    elif rnn_type == "gru":
        return nn.GRU(**common_kwargs)
    else:
        raise ValueError(f"Unknown rnn_type: {rnn_type!r}")


# -----------------------------------------------------------
# 1) Simple RNN model for next-step prediction (seq -> 1 value)
# -----------------------------------------------------------

class SimpleRNNModel(nn.Module):
    """
    Equivalent to your lecture SimpleRNNModel, but with:
      - configurable RNN type (RNN/LSTM/GRU)
      - config-driven hyperparameters

    Input:
        x: (batch, seq_len, input_size=1)

    Output:
        y: (batch, 1)  (predict next value)
    """

    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        rnn_type: RNNType = None,
        bidirectional: bool = None,
        dropout: float = None,
        output_size: int = None,
    ):
        super().__init__()

        # Defaults from config if not provided
        if input_size is None:
            input_size = cfg.model.input_size
        if hidden_size is None:
            hidden_size = cfg.model.hidden_size
        if num_layers is None:
            num_layers = cfg.model.num_layers
        if rnn_type is None:
            rnn_type = cfg.model.model_type
        if bidirectional is None:
            bidirectional = cfg.model.bidirectional
        if dropout is None:
            dropout = cfg.model.dropout
        if output_size is None:
            output_size = cfg.model.output_size

        self.rnn_type: RNNType = rnn_type
        self.bidirectional = bidirectional

        self.rnn = _make_rnn_module(
            rnn_type=rnn_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * num_directions, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_size)
        Returns:
            y : (batch, output_size)
        """
        # For LSTM, rnn returns (out, (h_n, c_n))
        out, hidden = self.rnn(x)

        # out: (batch, seq_len, hidden_size * num_directions)
        # We take the last time step
        last_out = out[:, -1, :]  # (batch, hidden_size * num_directions)

        y = self.fc(last_out)     # (batch, output_size)
        return y


# -----------------------------------------------------------
# 2) Sequence model: seq -> seq of predictions
# -----------------------------------------------------------

class RNNSequenceModel(nn.Module):
    """
    Similar to `RNNSequence` in your lecture code.

    Input:
        x: (batch, seq_len, input_size)

    Output:
        y: (batch, seq_len, output_size)

    This is useful if you want a prediction at each time step
    of the input sequence (or you can adapt it to output
    a different-length sequence if needed later).
    """

    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        rnn_type: RNNType = None,
        bidirectional: bool = None,
        dropout: float = None,
        output_size: int = None,
    ):
        super().__init__()

        # Defaults from config
        if input_size is None:
            input_size = cfg.model.input_size
        if hidden_size is None:
            hidden_size = cfg.model.hidden_size
        if num_layers is None:
            num_layers = cfg.model.num_layers
        if rnn_type is None:
            rnn_type = cfg.model.model_type
        if bidirectional is None:
            bidirectional = cfg.model.bidirectional
        if dropout is None:
            dropout = cfg.model.dropout
        if output_size is None:
            output_size = cfg.model.output_size

        self.rnn_type: RNNType = rnn_type
        self.bidirectional = bidirectional

        self.rnn = _make_rnn_module(
            rnn_type=rnn_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * num_directions, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_size)
        Returns:
            y : (batch, seq_len, output_size)
        """
        out, hidden = self.rnn(x)   # out: (batch, seq_len, hidden * num_directions)
        y = self.fc(out)            # (batch, seq_len, output_size)
        return y


# -----------------------------------------------------------
# 3) Model factory (optional but convenient)
# -----------------------------------------------------------

def build_model(sequence_output: bool = False) -> nn.Module:
    """
    Convenience function to build the appropriate model based on config.

    sequence_output = False:
        - Use SimpleRNNModel (seq -> 1 value).

    sequence_output = True:
        - Use RNNSequenceModel (seq -> seq of values).
    """
    if not sequence_output:
        model = SimpleRNNModel()
    else:
        model = RNNSequenceModel()

    return model.to(cfg.device)
