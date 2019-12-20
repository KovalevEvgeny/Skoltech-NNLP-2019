import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

def surround_nonalnum_with_spaces(text):
    cnt_nonalnum = 0
    for i in range(len(text)):
        j = i + 2 * cnt_nonalnum
        if not text[j].isalnum():
            text = text[:j] + ' ' + text[j] + ' ' + text[(j + 1):]
            cnt_nonalnum += 1
    return text

def preprocess(data):
    data_preprocessed = data.copy()
    data_preprocessed = [t.lower() for t in data_preprocessed]
    data_preprocessed = list(map(lambda t: surround_nonalnum_with_spaces(t), data_preprocessed))
    return data_preprocessed

def tokenize(data):
    return [t.split() for t in data]

def build_vocabulary(data):
    vocab = {'UNK': 0, 'PAD': 1}
    for d in data:
        for w in d:
            try:
                vocab[w]
            except:
                vocab[w] = len(vocab)
    return vocab

def build_emb_dict(file_path, vocab, d=300):
    emb_dict = dict()
    unk_array = np.zeros(d)
    with open(file_path, 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                vocab[word]
                vector = np.asarray(values[1:], "float32")
                emb_dict[word] = vector
                unk_array += vector
            except:
                continue
    emb_dict['UNK'] = unk_array / len(emb_dict)
    emb_dict['PAD'] = np.zeros(d)
    return emb_dict

def build_emb_matrix(file_path, vocab, d=300):
    emb_dict = build_emb_dict(file_path, vocab, d=d)
    emb_matrix = np.zeros((len(emb_dict), d))
    word2idx = {'UNK': 0, 'PAD': 1}
    for word in set(emb_dict.keys()) - set(['UNK', 'PAD']):
        word2idx[word] = len(word2idx)
    for w, i in word2idx.items():
        emb_matrix[i] = emb_dict[w]
    emb_matrix = torch.tensor(emb_matrix)
    return emb_matrix, word2idx

class BiLSTM(nn.Module):
    def __init__(self, emb_matrix, hidden_size=64, output_size=2, freeze_emb=True):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(emb_matrix)
        if freeze_emb:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x_emb = self.embedding(x)
        # (batch, seq_len, num_directions * hidden_size)
        lstm_out, _ = self.lstm(x_emb.float())
        # (batch, seq_len, num_directions, hidden_size)
        lstm_out = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], -1, self.hidden_size)
        # lstm_out[:, :, 0, :] -- output of the forward LSTM
        # lstm_out[:, :, 1, :] -- output of the backward LSTM
        # we take the last hidden state of the forward LSTM and the first hidden state of the backward LSTM
        x_fc = torch.cat((lstm_out[:, -1, 0, :], lstm_out[:, 0, 1, :]), dim=1)
        fc_out = self.fc(x_fc)
        out = self.softmax(fc_out)
        return out

def as_matrix(documents, word2idx, max_len=None):
    max_doc_len = max(map(len, documents))
    if max_len is None:
        max_len = max_doc_len
    else:
        max_len = min(max_doc_len, max_len)
    matrix = np.ones((len(documents), max_len), dtype=np.int64)
    for i, doc in enumerate(documents):
        row_ix = [word2idx.get(word, 0) for word in doc[:max_len]]
        matrix[i, :len(row_ix)] = row_ix
    return matrix

def train_bilstm(model, optimizer, train_data, train_labels_tensor, word2idx, max_len, device, n_epochs, batch_size):
    model.to(device)

    train_size = len(train_data)

    n_batches_train = (train_size - 1) // batch_size + 1

    for epoch in range(n_epochs):
        model.train()
        for i in range(0, train_size, batch_size):
            x = as_matrix(train_data[i:(i + batch_size)], word2idx, max_len)
            x = torch.tensor(x).long()
            y = train_labels_tensor[i:(i + batch_size)].float()
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = nn.BCELoss()(prediction, y)
            loss.backward()
            optimizer.step()
    return model

def set_random_seeds(seed_value=13, device='cpu'):
    '''source https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/5'''
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def predict_bilstm(model, dev_data, word2idx, max_len, device, batch_size=16):
    with torch.no_grad():
        val_size = len(dev_data)
        y_pred = np.zeros(val_size, dtype=float)
        for i in range(0, val_size, batch_size):
            x = as_matrix(dev_data[i:(i + batch_size)], word2idx, max_len)
            x = torch.tensor(x).long()
            x = x.to(device)
            prediction = model(x)[:, 1]
            y_pred[i:(i + batch_size)] = prediction.cpu().detach().numpy()
    return y_pred

def train(train_texts, train_labels, pretrain_params=None, EMBED_PATH='glove.6B.300d.txt', d=300, hidden_size=64, lr=0.05, max_len=300, epochs=84, batch_size=256, device='cuda'):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """
    train_labels_num = np.array([int(lab == 'pos') for lab in train_labels])
    train_labels_tensor = torch.tensor(train_labels_num)
    train_labels_tensor = torch.cat([1 - train_labels_tensor.view(-1, 1), train_labels_tensor.view(-1, 1)], dim=1)
    train_data = tokenize(preprocess(train_texts))
    train_vocab = build_vocabulary(train_data)
    emb_matrix, word2idx = build_emb_matrix(EMBED_PATH, train_vocab, d=d)
    
    bilstm = BiLSTM(emb_matrix, hidden_size)
    optimizer = optim.SGD(bilstm.parameters(), lr=lr)

    set_random_seeds(13, device)
    bilstm = train_bilstm(bilstm, optimizer, train_data, train_labels_tensor, word2idx, max_len=max_len, device=device, n_epochs=epochs, batch_size=batch_size)
    return {'bilstm': bilstm, 'word2idx': word2idx, 'max_len': max_len, 'device': device}

def pretrain(texts):
   """
   Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.
   :param texts: a list of texts (str objects), one str per example
   :return: learnt parameters, or any object you like (it will be passed to the train function)
   """
   ############################# PUT YOUR CODE HERE #######################################
   return None

def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """
    bilstm = params['bilstm']
    word2idx = params['word2idx']
    max_len = params['max_len']
    device = params['device']
    texts_preprocessed = tokenize(preprocess(texts))
    bilstm.to(device)
    bilstm.eval()
    y_pred = predict_bilstm(bilstm, texts_preprocessed, word2idx, max_len, device)
    y_pred = list(map(lambda y: 'pos' if y >= 0.5 else 'neg', y_pred))
    return y_pred