import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression
from learning_to_rank.utils import group_by, ndcg_k


class Net(nn.Module):
    def __init__(self, n_feature, h1_units, h2_units):
        super(Net, self).__init__()
        self.h1 = nn.Linear(n_feature, h1_units)

        self.h2 = nn.Linear(h1_units, h2_units)

        self.out = nn.Linear(h2_units, 1)

    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.out(x)
        return x
    def predict(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.out(x)
        n = x.data.numpy()[0]
        return n

class NRegression:
    def __init__(self, training_data, n_feature, h1_units = 512, h2_units = 256, epoch = 10, lr=0.001):
        self.training_data = training_data
        self.n_feature = n_feature
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.epoch = epoch
        self.learning_rate = lr
        self.model = Net(n_feature, h1_units, h2_units)


    def decay_learning_rate(self, optimizer, epoch, decay_rate):
        if (epoch+1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay_rate

    def fit(self):
        net = self.model
        qid_doc_map = group_by(self.training_data, 1)
        query_idx = qid_doc_map.keys()
        # true_scores is a matrix, different rows represent different queries
        tmp = [self.training_data[qid_doc_map[qid], 0] for qid in query_idx]
        true_scores = []
        for i in tmp:
            true_scores += (list(i))
        true_scores = torch.Tensor(true_scores).reshape(-1,1)

        sample_num = len(self.training_data)
        loss_fn = nn.MSELoss()
        loss_list = []
        optimizer = optim.Adam(net.parameters(),lr=self.learning_rate)
        print('Training .....')
        for i in range(self.epoch):
            predicted_scores = self.model(torch.from_numpy(self.training_data[:, 2:].astype(np.float32)))

            self.decay_learning_rate(optimizer, i, 0.95)
            net.zero_grad()
            loss = loss_fn(predicted_scores,true_scores)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data.numpy())
            if i % 5 == 0:
                print('Epoch:{}, loss : {}'.format(i, loss.item()))
    def validate(self, test_data, k):
        """
        compute the average NDCG@k for the given test data.
        :param test_data: test data
        :param k: used to compute NDCG@k
        :return:
        """
        # load model parameters
        net = Net(self.n_feature, self.h1_units, self.h2_units)
        # net.load_state_dict(torch.load('ranknet.pkl'))

        qid_doc_map = group_by(test_data, 1)
        query_idx = qid_doc_map.keys()
        ndcg_k_list = []

        for q in query_idx:
            true_scores = test_data[qid_doc_map[q], 0]
            if sum(true_scores) == 0:
                continue
            docs = test_data[qid_doc_map[q]]
            X_test = docs[:, 2:]

            predicted_scores = [net.predict(torch.Tensor(test_x).data) for test_x in X_test]
            pred_rank = np.argsort(predicted_scores)[::-1]
            pred_rank_score = true_scores[pred_rank]
            ndcg_val = ndcg_k(pred_rank_score, k)
            ndcg_k_list.append(ndcg_val)
        average_ndcg = np.nanmean(ndcg_k_list)
        return average_ndcg, predicted_scores