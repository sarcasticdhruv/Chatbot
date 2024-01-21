import torch
import torch.nn as nn
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

# Define the text fields
TEXT = Field(tokenize='spacy', tokenizer_language='en', lower=True, init_token='<sos>', eos_token='<eos>', include_lengths=True)
LABEL = Field(sequential=True, use_vocab=False, pad_token='<pad>', dtype=torch.float)

# Load the dataset
train_data, valid_data, test_data = Multi30k.splits(exts=('.en', '.de'), fields=(TEXT, LABEL))

# Build the vocabulary
TEXT.build_vocab(train_data, min_freq=2)
LABEL.build_vocab(train_data)

# Define the LSTM model
class ChatbotModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers):
        super(ChatbotModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_state = None
        self.cell_state = None

    def forward(self, x):
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
            self.cell_state = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)

        embed = self.embedding(x)
        output, (hidden_state, cell_state) = self.lstm(embed, (self.hidden_state, self.cell_state))
        self.hidden_state = hidden_state
        self.cell_state = cell_state
        output = self.fc(output[:, -1, :])
        return output

# Initialize the model
input_dim = len(TEXT.vocab)
embedding_dim = 256
hidden_dim = 256
output_dim = len(LABEL.vocab)
n_layers = 2
model = ChatbotModel(input_dim, embedding_dim, hidden_dim, output_dim, n_layers)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create an iterator for the training data
train_iterator = BucketIterator(train_data, batch_size=32, sort_key=lambda x: len(x.text), repeat=False)

# Train the model
for epoch in range(10):
    for i, batch in enumerate(train_iterator):
        src = batch.text
        trg = batch.label
        optimizer.zero_grad()
        output = model(src)
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, :-1].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

# Prompt the user for input
prompt = input("You: ")

# Tokenize the input
tokens = [TEXT.init_token] + TEXT.tokenize(prompt) + [TEXT.eos_token]

# Convert the tokens to tensor
tensor = torch.tensor([TEXT.vocab[token] for token in tokens], dtype=torch.long)

# Add a batch dimension
tensor = tensor.unsqueeze(0)

# Generate the response
with torch.no_grad():
    output = model(tensor)

# Get the most likely word
_, predicted_token = torch.max(output, dim=1)

# Convert the predicted word back to a token
predicted_word = TEXT.vocab[predicted_token]

# Print the response
print(f"Bot: {predicted_word}")