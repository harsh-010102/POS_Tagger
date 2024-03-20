import torch
from nltk import word_tokenize
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from conllu import parse_incr
import nltk
from sklearn.preprocessing import OneHotEncoder
nltk.download('punkt')

import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

p = 4
s = 4
embedding_dim = 200
num_of_hidden_layer = 5
layer_size = 60

def read_file(file_path):
        pos_tags_set = []  # Initialize set for unique POS tags
        sentences = []  # Initialize list for sentences
        with open(file_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                tokens = [token['form'] for token in tokenlist]
                pos_tags = [token['upostag'] for token in tokenlist]
                for i in pos_tags:
                    pos_tags_set.append(i)

                sentences.append((tokens, pos_tags))
        return pos_tags_set, sentences

def preprocess(sentence_list, p, s):
    updated_list = []
    for sentence, tags in sentence_list:
        if sentence:
            temp = ['<s>'] * p
            temp1 = ['</s>'] * s
            updated_sentence = temp + sentence + temp1
            updated_list.append((updated_sentence, tags))
    return updated_list

pos_tags_set_train, training_data = read_file(r'en_atis-ud-train.conllu')
pos_tags_set_val, validation_data = read_file(r'en_atis-ud-dev.conllu')
pos_tags_set_test, test_data = read_file(r'en_atis-ud-test.conllu')

training_data = preprocess(training_data, p, s)
validation_data = preprocess(validation_data, p, s)
test_data = preprocess(test_data, p, s)
count = 0
pos_tags_set_train_dict = {}
for pos in pos_tags_set_train:
    if pos in pos_tags_set_train_dict:
        continue
    pos_tags_set_train_dict[pos] = count
    count += 1
# print(pos_tags_set_train_dict)    
# print(pos_tags_set_train_dict)

word_freq_of_train = dict()
for sentence, tags in training_data:
    for word in sentence:
        if word in word_freq_of_train:
            word_freq_of_train[word] += 1
        else:
            word_freq_of_train[word] = 1
train_vocab = dict()
train_vocab['<s>'] = 0
train_vocab['</s>'] = 1
train_vocab['<unk>'] = 2
for word in word_freq_of_train:
    if word_freq_of_train[word] >= 3 and word not in train_vocab:
        train_vocab[word] = len(train_vocab)
# print(train_vocab)

def get_context_tag_pair(data, p, s):
    context_li = []

    for words, tag_sequence in data:
        for i in range(p, len(words) - s):
            if tag_sequence[i-p] == 'SYM':
                # print("Yes")
                continue
            context = words[i - p : i + s + 1]
            context_li.append([context, tag_sequence[i-p]])
    return context_li

context_li_train = get_context_tag_pair(training_data, p, s)

context_li_val = get_context_tag_pair(validation_data, p, s)

# print(len(context_li))
context_li_test = get_context_tag_pair(test_data, p, s)
# print(len(context_li))
# print(context_li_test[0:16])
# print(len(context_li_test))


def get_one_hot_encoding(pos_tags):
    pos_tags_embeddings = [[1.0 if i == j else 0.0 for j in range(len(pos_tags))] for i in range(len(pos_tags))]

    return pos_tags_embeddings
pos_tags_set_train = list(pos_tags_set_train_dict.keys())
one_hot_embedding_of_pos = get_one_hot_encoding(pos_tags_set_train)
one_hot_embedding_of_pos_dict = dict()
for i in range(len(pos_tags_set_train)):
    one_hot_embedding_of_pos_dict[pos_tags_set_train[i]] = one_hot_embedding_of_pos[i]
idx_to_pos = dict()
for key, val in pos_tags_set_train_dict.items():
    idx_to_pos[val] = key
# pos_tags_set_train = list(pos_tags_set_train)
# one_hot_embedding_of_pos = get_one_hot_encoding(pos_tags_set_train)
# one_hot_embedding_of_pos_dict = dict()
# for i in range(len(pos_tags_set_train)):
#     one_hot_embedding_of_pos_dict[pos_tags_set_train[i]] = one_hot_embedding_of_pos[i]
# print(one_hot_embedding_of_pos_dict)

class POSDataset(Dataset):
    def __init__(self, vocab, pos, context):
        super().__init__()
        self.vocab = vocab
        self.pos = pos
        self.context = context

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        word_context = []
        for word in self.context[idx][0]:
            if word in self.vocab:
                word_context.append(self.vocab[word])
            else:
                word_context.append(self.vocab['<unk>'])
        pos_tag = self.context[idx][1]
        pos_emb = self.pos[pos_tag]
        return torch.tensor(word_context), torch.tensor(pos_emb)

train_dataset = POSDataset(train_vocab, one_hot_embedding_of_pos_dict, context_li_train)
train_loader = DataLoader(train_dataset, batch_size=32)

val_dataset = POSDataset(train_vocab, one_hot_embedding_of_pos_dict, context_li_val)
val_loader = DataLoader(val_dataset, batch_size=32)

test_dataset = POSDataset(train_vocab, one_hot_embedding_of_pos_dict, context_li_test)
test_loader = DataLoader(test_dataset, batch_size=32)

class POS_Tagger(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, output_size, num_of_hidden_layer, layer_size):
        super().__init__()
        self.embedding_module = torch.nn.Embedding(vocabulary_size, embedding_dim)
        h_layer = []

        h_layer.append(torch.nn.Linear(embedding_dim*(p+1+s), layer_size))
        h_layer.append(torch.nn.Tanh())

        for i in range(num_of_hidden_layer-1) :
            h_layer.append(torch.nn.Linear(layer_size, layer_size))
            h_layer.append(torch.nn.Tanh())
        h_layer.append(torch.nn.Linear(layer_size, output_size))
        self.model = torch.nn.Sequential(*h_layer) 

    def forward(self, word_index: torch.Tensor):
        embedding = self.embedding_module(word_index)
        flattened = embedding.view(-1, embedding_dim * (p + 1 + s))
        return self.model(flattened)

    def predict(self, word_index):
        with torch.no_grad():
            output = self.forward(word_index)
            return torch.argmax(output, dim=1)

ffnn_model = POS_Tagger(len(train_vocab), embedding_dim, len(pos_tags_set_train), num_of_hidden_layer, layer_size)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ffnn_model.parameters(), lr=0.001)

for epoch_num in range(10):
  ffnn_model.train()
  error = 0
  for batch_num, (words, tags) in enumerate(train_loader):
    pred = ffnn_model(words)

    loss = loss_fn(pred, tags)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    error += loss.item()
  print(f"{epoch_num=}", " Loss : ",  error)

def evaluate(ffnn_model, data):
    ffnn_model.eval()

    total_correct = 0
    total_samples = 0
    predicted_val_list = []
    target_tag_values_list = []
    with torch.no_grad():
        for batch_num, (words, tags) in enumerate(data):
            pred = ffnn_model(words)
            _, predicted_labels = torch.max(pred, 1)

            tags_values = torch.argmax(tags, dim=1)

            total_correct += (predicted_labels == tags_values).sum().item()
            total_samples += len(tags)

            for i in range(len(predicted_labels)):
                predicted_val_list.append(predicted_labels[i])
            for i in range(len(tags_values)):
                target_tag_values_list.append(tags_values[i])

    accuracy = total_correct / total_samples
    print(f"Accuracy on dataset: {accuracy}")
    return accuracy, predicted_val_list, target_tag_values_list

dev_accuracy, dev_predicted_val_list, dev_target_tag_values_list = evaluate(ffnn_model, val_loader)
# print(len(dev_predicted_val_list[2]))
# print(len(dev_target_tag_values_list[2]))

test_accuracy, test_predicted_val_list, test_target_tag_values_list = evaluate(ffnn_model, test_loader)

print("Dev accuracy = ", dev_accuracy)
print("Test accuracy = ", test_accuracy)

torch.save(ffnn_model.state_dict(), 'model.pt')

# loaded_model  = POS_Tagger(len(train_vocab), embedding_dim, len(pos_tags_set_train))
# loaded_model.load_state_dict(torch.load('model.pt'))
# print(loaded_model)

from nltk.tokenize import word_tokenize
def get_key_from_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None  # Return None if the value is not found in the dictionary


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
dev_accuracy = accuracy_score(dev_target_tag_values_list, dev_predicted_val_list)
dev_classification_report = classification_report(dev_target_tag_values_list, dev_predicted_val_list)
dev_confusion_matrix = confusion_matrix(dev_target_tag_values_list, dev_predicted_val_list)

print("Dev Accuracy:", dev_accuracy)

print("Dev Classification Report:\n", dev_classification_report)
print("Dev Confusion Matrix:\n", dev_confusion_matrix)

test_accuracy = accuracy_score(test_target_tag_values_list, test_predicted_val_list)
test_classification_report = classification_report(test_target_tag_values_list, test_predicted_val_list)
test_confusion_matrix = confusion_matrix(test_target_tag_values_list, test_predicted_val_list)


print("Test Accuracy:", test_accuracy)
print("Test Classification Report:\n", test_classification_report)
print("Test Confusion Matrix:\n", test_confusion_matrix)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

# Accuracy
dev_accuracy = accuracy_score(dev_target_tag_values_list, dev_predicted_val_list)
test_accuracy = accuracy_score(test_target_tag_values_list, test_predicted_val_list)

# Classification report (includes precision, recall, F1-score)
dev_report = classification_report(dev_target_tag_values_list, dev_predicted_val_list, output_dict=True)
test_report = classification_report(test_target_tag_values_list, test_predicted_val_list, output_dict=True)

# Calculate precision, recall, f1-score using confusion matrix
dev_precision, dev_recall, dev_f1, _ = precision_recall_fscore_support(dev_target_tag_values_list, dev_predicted_val_list, average='micro')
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_target_tag_values_list, test_predicted_val_list, average='micro')

dev_precision_macro, dev_recall_macro, dev_f1_macro, _ = precision_recall_fscore_support(dev_target_tag_values_list, dev_predicted_val_list, average='macro')
test_precision_macro, test_recall_macro, test_f1_macro, _ = precision_recall_fscore_support(test_target_tag_values_list, test_predicted_val_list, average='macro')

# Confusion matrices
dev_conf_matrix = confusion_matrix(dev_target_tag_values_list, dev_predicted_val_list)
test_conf_matrix = confusion_matrix(test_target_tag_values_list, test_predicted_val_list)

# Print or visualize the results
print("Development Set Metrics:")
print("Accuracy:", dev_accuracy)
print("Precision (Micro) on Development Set:", dev_precision)
print("Recall (Micro) on Development Set:", dev_recall)
print("F1-score (Micro) on Development Set:", dev_f1)
print("Precision (Macro) on Development Set:", dev_precision_macro)
print("Recall (Macro) on Development Set:", dev_recall_macro)
print("F1-score (Macro) on Development Set:", dev_f1_macro)
# print("Confusion Matrix:\n", dev_conf_matrix)

print("\nTest Set Metrics:")
print("Accuracy:", test_accuracy)
print("Precision (Micro) on Test Set:", test_precision)
print("Recall (Micro) on Test Set:", test_recall)
print("F1-score (Micro) on Test Set:", test_f1)
print("Precision (Macro) on Test Set:", test_precision_macro)
print("Recall (Macro) on Test Set:", test_recall_macro)
print("F1-score (Macro) on Test Set:", test_f1_macro)
# print("Confusion Matrix:\n", test_conf_matrix)


import json

try:
    with open('ffnn_acc.txt', 'r') as f:
        data = f.read()
        if data:  
            my_dict = json.loads(data)  
        else:
            my_dict = {}
except FileNotFoundError:
    my_dict = {}

# Adding or updating the dictionary with new key-value pair

accuracy_value = dev_accuracy  
my_dict[str(p)] = accuracy_value
print(my_dict)

# sorted_data = dict(sorted(my_dict.items()))

# Print the sorted dictionary

# Writing the updated dictionary back to the file
with open('ffnn_acc.txt', 'w') as f:
    json.dump(my_dict, f)
keys = list(my_dict.keys())
values = list(my_dict.values())

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(keys, values, marker='o')
plt.xlabel('Context')
plt.ylabel('Accuracy')
plt.title('Context vs Accuracy')
plt.grid()
plt.xticks() 
plt.show()
