import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.elmo import Elmo, batch_to_ids

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

class BiLSTM_ELMo(nn.Module):
    def __init__(self, elmo, input_size=512, hidden_size=64, output_size=2):
        super(BiLSTM_ELMo, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = elmo
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x_emb = self.embedding(x)['elmo_representations'][0]
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

def train_bilstm(model, optimizer, train_data, train_labels_tensor, max_len, device, n_epochs, batch_size):
    model.to(device)

    train_size = len(train_data)

    n_batches_train = (train_size - 1) // batch_size + 1

    for epoch in range(n_epochs):
        model.train()
        for i in range(0, train_size, batch_size):
            x = batch_to_ids(train_data[i:(i + batch_size)])
            if max_len is not None:
                x = x[:, :max_len]
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

def predict_bilstm(model, dev_data, max_len, device, batch_size=1):
    with torch.no_grad():
        val_size = len(dev_data)
        y_pred = np.zeros(val_size, dtype=float)
        for i in range(0, val_size, batch_size):
            x = batch_to_ids(dev_data[i:(i + batch_size)])
            if max_len is not None:
                x = x[:, :max_len]
            x = x.to(device)
            prediction = model(x)[:, 1]
            y_pred[i:(i + batch_size)] = prediction.cpu().detach().numpy()
    return y_pred

def train(
    train_texts,
    train_labels,
    pretrain_params=None,
    options_file="elmo_2x2048_256_2048cnn_1xhighway_options.json",
    weight_file="elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5",
    hidden_size=64,
    lr=0.1,
    max_len=150,
    epochs=5,
    batch_size=64,
    device='cuda'
    ):
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
    
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    bilstm_elmo = BiLSTM_ELMo(elmo, hidden_size=hidden_size)
    optimizer = optim.SGD(bilstm_elmo.parameters(), lr=lr)

    set_random_seeds(13, device)
    bilstm = train_bilstm(bilstm_elmo, optimizer, train_data, train_labels_tensor, max_len=max_len, device=device, n_epochs=epochs, batch_size=batch_size)
    return {'bilstm': bilstm, 'max_len': max_len, 'device': device}

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
    max_len = params['max_len']
    device = params['device']
    texts_preprocessed = tokenize(preprocess(texts))
    bilstm.to(device)
    bilstm.eval()
    y_pred = predict_bilstm(bilstm, texts_preprocessed, max_len, device)
    y_pred = list(map(lambda y: 'pos' if y >= 0.5 else 'neg', y_pred))
    return y_pred