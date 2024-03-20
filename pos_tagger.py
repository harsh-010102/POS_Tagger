import torch
from nltk.tokenize import word_tokenize
from conllu import parse_incr
import sys
# print(len(sys.argv))
if(len(sys.argv) != 2):
    print("Invalid command")
    exit(1)
model_type = sys.argv[1]
if(model_type == '-f'):
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
    training_data = preprocess(training_data, p, s)
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
    
    def get_one_hot_encoding(pos_tags):
        pos_tags_embeddings = [[1.0 if i == j else 0.0 for j in range(len(pos_tags))] for i in range(len(pos_tags))]

        return pos_tags_embeddings

    pos_tags_set_train = list(pos_tags_set_train_dict.keys())
    # print(pos_tags_set_train)
    one_hot_embedding_of_pos = get_one_hot_encoding(pos_tags_set_train)
    one_hot_embedding_of_pos_dict = dict()
    for i in range(len(pos_tags_set_train)):
        one_hot_embedding_of_pos_dict[pos_tags_set_train[i]] = one_hot_embedding_of_pos[i]
        
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
    ffnn_model.load_state_dict(torch.load("model.pt"))
    idx_to_pos = dict()
    for key, val in pos_tags_set_train_dict.items():
        idx_to_pos[val] = key
    sentence = input("Enter sentence : ")
    words = word_tokenize(sentence)
    punc = [',', '.', '!', '?']
    lowercase_words = [word.lower() for word in words if word not in punc]

    sentence_words = lowercase_words
    # print(sentence_words)

    def preprocess_sentence(words, p, s):
        temp = ['<s>'] * p
        temp1 = ['</s>'] * s
        updated_sent = temp + words + temp1
        return updated_sent
    words = preprocess_sentence(sentence_words, p, s)
    # print(words)
    output_pos_tag = []
    for i in range(p, len(words) - s):
        word_idx = []
        context = words[i - p : i + s + 1]
        for word in context:
            if word in train_vocab:
                idx = train_vocab[word]
            else:
                idx = train_vocab['<unk>']
            word_idx.append(idx)
        # print(word_idx)
        predictions = ffnn_model.predict(torch.tensor(word_idx))
        # print(idx_to_pos[predictions.item()])
        output_pos_tag.append(idx_to_pos[predictions.item()])
    for i in range(len(output_pos_tag)):
        print(sentence_words[i], output_pos_tag[i])

else:
    embedding_dim = 200
    num_layers = 4
    bidirectional = True
    hidden_dim = 10
    
    def read_file(file_path):
        pos_tags_set = []  
        sentences = [] 
        with open(file_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                tokens = [token['form'] for token in tokenlist]
                pos_tags = [token['upostag'] for token in tokenlist]
                for i in pos_tags:
                    pos_tags_set.append(i)
                sentences.append((tokens, pos_tags))
        return pos_tags_set, sentences


    pos_tags_set_train, training_data = read_file(r'C:\Users\Harsh\Desktop\NLP\Assignment-2\en_atis-ud-train.conllu')

    count = 0
    pos_tags_set_train_dict = {}
    for pos in pos_tags_set_train:
        if pos in pos_tags_set_train_dict:
            continue
        pos_tags_set_train_dict[pos] = count
        count += 1
    idx_to_pos = dict()
    for key, val in pos_tags_set_train_dict.items():
        idx_to_pos[val] = key
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
    train_vocab['<pad>'] = 3

    for word in word_freq_of_train:
        if word_freq_of_train[word] >= 3 and word not in train_vocab:
            train_vocab[word] = len(train_vocab)
    # print(train_vocab)

    def get_context_tag_pair(data, p, s):
        context_li = []
        for words, tag_sequence in data:
            word = []
            tag = []
            for i in range(len(words)):
                if tag_sequence[i] == 'SYM':
                    # print('yes')
                    continue
                word.append(words[i])
                tag.append(tag_sequence[i])
            context_li.append([word, tag])
        return context_li

    context_li_train = get_context_tag_pair(training_data, 3, 3)


    # print(context_li_test[0][0])

    class LSTMTagger(torch.nn.Module):
        def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size, num_layers=1, bidirectional=False):
            super(LSTMTagger, self).__init__()
            
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            
            self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

            self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional)
            
            if bidirectional:
                self.hidden2tag = torch.nn.Linear(hidden_dim * 2, target_size)
            else:
                self.hidden2tag = torch.nn.Linear(hidden_dim, target_size)
            
        def forward(self, sentence):
            embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
            tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
            return tag_scores
        
        def predict(self, sentence):
            with torch.no_grad():
                output = self.forward(sentence)
                return torch.argmax(output, dim=1)


    # print(context_li_test[0][1])
    # print(len(context_li_test))
    pos_tags_set_train = list(pos_tags_set_train_dict.keys())
    rnn_model = LSTMTagger(embedding_dim, hidden_dim, len(train_vocab), len(pos_tags_set_train), num_layers, bidirectional)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    rnn_model.load_state_dict(torch.load("model2.pt"))

    sentence = input("Enter sentence : ")
    words = word_tokenize(sentence)
    punc = [',', '.', '!', '?']
    lowercase_words = [word.lower() for word in words if word not in punc]
    words = lowercase_words
    output_pos_tag = []
    word_idx = []
    for word in words:
        if word in train_vocab:
            idx = train_vocab[word]
        else:
            idx = train_vocab['<unk>']
        word_idx.append(idx)

        # print(word_idx)
    predictions = rnn_model.predict(torch.tensor(word_idx))
    for i in predictions:
        output_pos_tag.append(idx_to_pos[i.item()])

    for i in range(len(words)):
        print(words[i], output_pos_tag[i])