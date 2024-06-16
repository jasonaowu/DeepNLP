import os
import re
import sys
from collections import Counter
from tqdm import tqdm
import jieba
import torch
from torch import nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
import matplotlib.pyplot as plt
def preprocess_text(text):
    """Preprocess the text by removing unwanted characters and symbols."""
    text = re.sub('----〖新语丝电子文库\(www.xys.org\)〗', '', text)
    text = re.sub('本书来自www.cr173.com免费txt小说下载站', '', text)
    text = re.sub('更多更新免费电子书请关注www.cr173.com', '', text)
    text = re.sub('\u3000', '', text)
    text = re.sub(r'[。，、；：？！（）《》【】“”‘’…—\-,.:;?!\[\](){}\'"<>]', '', text)
    text = re.sub(r'[\n\r\t]', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return text
def load_data(directory, limit=4):
    """Load data from the specified directory, limited to a certain number of files."""
    corpus = []
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='ansi') as file:
                corpus.append(file.read())
        if i + 1 == limit:
            break
    return corpus
# Define your directory path where data files are located
directory = r'data/'

# Load data into corpus
corpus = load_data(directory)

# Now you can use corpus for further processing
words = [word for text in corpus for word in jieba.lcut(text)]
print(len(words))

# Tokenization using jieba
words = [word for text in corpus for word in jieba.lcut(text)]

# Build vocabulary
counter = Counter(words)
counter['<unk>'] = 0
vocab = Vocab(counter)
vocab_size = len(vocab)
directory = r'data/'
corpus = load_data(directory)




tokenizer = get_tokenizer('basic_english')

# Convert text to sequences of indices
words_str = ' '.join(words)
tokens = tokenizer(words_str)
sequences = [vocab[token] for token in tokens]
sequences = [word if word < vocab_size else vocab['<unk>'] for word in sequences]
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_units, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, (h, c) = self.encoder(x)
        out, _ = self.decoder(x, (h, c))
        out = self.fc(out)
        return out

# Training loop
embedding_dim = 256
hidden_units = 50
model = Seq2Seq(vocab_size, embedding_dim, hidden_units)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Ensure sequences fit within vocabulary size
assert max(sequences) < vocab_size, "Vocabulary size is insufficient to encode all words."

# Training loop
for epoch in tqdm(range(10)):
    optimizer.zero_grad()
    output = model(torch.tensor(sequences[:1000]))
    loss = criterion(output, torch.tensor(sequences[:1000]))
    loss.backward()
    optimizer.step()

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_units, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, (h, c) = self.encoder(x)
        out, _ = self.decoder(x, (h, c))
        out = self.fc(out)
        return out

# Training loop
embedding_dim = 256
hidden_units = 50
model = Seq2Seq(vocab_size, embedding_dim, hidden_units)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters())

# Define the learning rate
learning_rate = 0.001

# Create the optimizer with specified learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Ensure sequences fit within vocabulary size
assert max(sequences) < vocab_size, "Vocabulary size is insufficient to encode all words."

# Lists to store losses for plotting
train_losses = []
epoch = 10
# Training loop
for epoch in tqdm(range(epoch)):
    optimizer.zero_grad()
    output = model(torch.tensor(sequences[:1000]))
    loss = criterion(output, torch.tensor(sequences[:1000]))
    loss.backward()
    optimizer.step()
    
    # Append current loss to list
    train_losses.append(loss.item())

    # Print the loss (optional)
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

# Plotting the training loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(0, epoch+1), train_losses, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


torch.save(model.state_dict(), 'model.pth')

model = Seq2Seq(vocab_size, embedding_dim, hidden_units)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Text generation setup
# start_text = "张无忌快步走近山脚，正要上峰，忽见山道旁中白光微闪，有人执着兵刃埋伏。他急忙停步，只过得片刻，见树丛中先后窜出四人，三前一后，齐向峰顶奔去。"
start_text = "乔峰来姑苏，本是找慕容复查清丐帮副帮主马大元被他自己的成名绝技所杀一事，谁知帮内突生大变，他被指证为契丹人。为解开自己的身世之谜，他北上少室山， 找自己的养父乔三槐和恩师玄苦，可二人已遇害身亡，目击之人皆认为是乔峰所为。"
start_words = list(jieba.cut(start_text))

word2idx = {word: idx for idx, word in enumerate(counter)}
idx2word = {idx: word for idx, word in enumerate(counter)}

# Convert starting text to sequence of indices
start_sequence = [word2idx[word] for word in start_words if word in word2idx]
if not start_sequence:
    raise ValueError("Start sequence is empty. Please provide a non-empty start sequence.")
input = torch.tensor(start_sequence).long().unsqueeze(0)

# Generate text
max_length = 50
generated_sequence = []

for _ in range(max_length):
    output = model(input)
    output_mean = output.mean(dim=1)
    next_word_idx = output_mean.argmax().item()
    generated_sequence.append(next_word_idx)
    input = torch.tensor([next_word_idx]).unsqueeze(0)

# Convert indices back to words
generated_words = [idx2word[idx] for idx in generated_sequence]
generated_text = ''.join(generated_words)

print(generated_text)
