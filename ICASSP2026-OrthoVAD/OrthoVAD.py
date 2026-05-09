import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class OrthoVAD(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, dropout_rate=0.3):
        super(OrthoVAD, self).__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        gru_output_dim = hidden_dim * 2

        self.temporal_attention = nn.Sequential(
            nn.Linear(gru_output_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.gru_layer_norm = nn.LayerNorm(gru_output_dim)

        self.sparsity_gate = nn.Sequential(
            nn.Linear(gru_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, gru_output_dim),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(gru_output_dim, 512)
        self.fc_layer_norm = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        self.apply(weights_init)

    def forward(self, inputs, seq_len):
        packed_x = pack_padded_sequence(
            inputs,
            seq_len.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_gru_out, _ = self.gru(packed_x)
        gru_out, _ = pad_packed_sequence(packed_gru_out, batch_first=True)

        attention_scores = self.temporal_attention(gru_out)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_aware_features = gru_out * attention_weights

        gate_values = self.sparsity_gate(context_aware_features)
        refined_features = context_aware_features * gate_values

        x = self.gru_layer_norm(refined_features)

        x = self.fc1(x)
        x = self.fc_layer_norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)

        scores = self.sigmoid(self.classifier(x))

        return scores, refined_features, context_aware_features, gate_values


# Backward-compatible alias for older scripts/checkpoints that import MODEL.
MODEL = OrthoVAD
