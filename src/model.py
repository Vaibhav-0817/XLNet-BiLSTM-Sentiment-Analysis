import torch.nn as nn
from transformers import XLNetForSequenceClassification

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.xlnet = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):
        xlnet_output = self.xlnet(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pooled_output = xlnet_output.hidden_states[-1][:, 0, :]
        lstm_output, _ = self.bilstm(pooled_output.unsqueeze(0))
        lstm_output = lstm_output.squeeze(0)
        lstm_output = self.dropout(lstm_output)
        output = self.fc(lstm_output)
        return output
