import torch.nn as nn

class NERModel(nn.Module):
    def __init__(self, bert_model, config, num_labels):
        super(NERModel, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        logits = self.fc(last_hidden_state)
        return logits
