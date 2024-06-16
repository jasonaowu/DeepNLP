# Import necessary libraries
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Check if CUDA (GPU support) is available
print(torch.cuda.is_available())

# Function to load stopwords from a file
def load_stopwords(file_path):
    stop_words = []
    with open(file_path, "r", encoding="gb18030", errors="ignore") as f:
        stop_words.extend([word.strip('\n') for word in f.readlines()])
    return stop_words

# Function to preprocess the text corpus by removing stopwords
def preprocess_corpus(text, cn_stopwords):
    for tmp_char in cn_stopwords:
        text = text.replace(tmp_char, "")
    return text 

# Initialize variables
merged_content = ''
stopwords_file_path = 'cn_stopwords.txt'
cn_stopwords = load_stopwords(stopwords_file_path)
book_titles_list = "白马啸西风,碧血剑,飞狐外传,连城诀,鹿鼎记,三十三剑客图,射雕英雄传,神雕侠侣,书剑恩仇录,天龙八部,侠客行,笑傲江湖,雪山飞狐,倚天屠龙记,鸳鸯刀,越女剑"

# Load and merge content from all books
for book_title in book_titles_list.split(','):
    book_title = book_title.strip()
    file_path = './data/{}.txt'.format(book_title)
    with open(file_path, 'r', encoding='gb18030') as f:
        merged_content += f.read()

# Preprocess the merged content by removing stopwords
merged_content = preprocess_corpus(merged_content, cn_stopwords)

# Save the preprocessed content to a new file (UTF-8 encoding)
output_file_path = 'all_utf8.txt'
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(merged_content)

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set pad token to eos token
tokenizer.pad_token = tokenizer.eos_token


# Create dataset using the Datasets library
datasets = load_dataset('text', data_files={'train': output_file_path})
tokenized_datasets = datasets.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128), batched=True)

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments
# training_args = TrainingArguments(
#     output_dir="./gpt2_jin_yong",
#     overwrite_output_dir=True,
#     num_train_epochs=1,
#     per_device_train_batch_size=4,
#     save_steps=10_000,
#     save_total_limit=2,
#     learning_rate=5e-5,
#     weight_decay=0.01,
# )
training_args = TrainingArguments(
    output_dir="./gpt2_jin_yong",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
)
# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
trainer.save_model("./gpt2_jin_yong")
tokenizer.save_pretrained("./gpt2_jin_yong")

# Load the saved model and tokenizer
output_dir = "./gpt2_jin_yong"
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model = GPT2LMHeadModel.from_pretrained(output_dir)

# Function to generate text using the trained model
def generate_text_transformer(seed_text, next_words, model, tokenizer):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=next_words + len(input_ids[0]), num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Generate a sample text
sample_text = "张无忌快步走近山脚，正要上峰，忽见山道旁中白光微闪，有人执着兵刃埋伏。他急忙停步，只过得片刻，见树丛中先后窜出四人，三前一后，齐向峰顶奔去。"
print(generate_text_transformer(sample_text, 100, model, tokenizer))
