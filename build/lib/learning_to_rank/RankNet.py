import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from learning_to_rank.utils import group_by, get_pairs, compute_lambda, ndcg_k, split_pairs

class Model(nn.Module):
    """
    construct the RankNet
    """
    def __init__(self, n_feature, h1_units, h2_units):
        super(Model, self).__init__()

        self.model = torch.nn.Sequential(
            # h_1
            torch.nn.Linear(n_feature, h1_units),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            # h_2
            torch.nn.Linear(h1_units, h2_units),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            # output
            torch.nn.Linear(h2_units, 1),
        )
        self.output_sig = torch.nn.Sigmoid()

    def forward(self, input_1, input_2):
        s1 = self.model(input_1)
        s2 = self.model(input_2)
        out = self.output_sig(s1-s2)
        return out

    def predict(self, input_):
        s = self.model(input_)
        n = s.data.numpy()[0]
        return n


class RankNet():
    """
    user interface
    """
    def __init__(self, training_data ,n_feature, h1_units = 512, h2_units = 256, epoch = 10, learning_rate = 0.01):
        self.training_data = training_data
        self.n_feature = n_feature
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.model = Model(n_feature, h1_units, h2_units)
        self.epoch = epoch
        self.learning_rate = learning_rate

    def decay_learning_rate(self, optimizer, epoch, decay_rate):
        if (epoch+1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay_rate

    def fit(self):
        """
        train the RankNet based on training data.
        After training, save the parameters of RankNet, named 'parameters.pkl'
        :param training_data:
        """

        net = self.model
        qid_doc_map = group_by(self.training_data, 1)
        query_idx = qid_doc_map.keys()
        # true_scores is a matrix, different rows represent different queries
        true_scores = [self.training_data[qid_doc_map[qid], 0] for qid in query_idx]

        order_paris = []
        for scores in true_scores:
            order_paris.append(get_pairs(scores))

        relevant_doc, irrelevant_doc = split_pairs(order_paris ,true_scores)
        relevant_doc = self.training_data[relevant_doc]
        irrelevant_doc = self.training_data[irrelevant_doc]

        X1 = relevant_doc[:, 2:]
        X2 = irrelevant_doc[:, 2:]
        y = np.ones((X1.shape[0], 1))

        # training......
        X1 = torch.Tensor(X1)
        X2 = torch.Tensor(X2)
        y = torch.Tensor(y)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)

        loss_fun = torch.nn.BCELoss()

        loss_list = []


        print('Train the model RankNet')
        for i in range(self.epoch):
            self.decay_learning_rate(optimizer, i, 0.95)

            net.zero_grad()
            y_pred = net(X1, X2)
            loss = loss_fun(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data.numpy())
            if i % 5 == 0:
                print('Epoch:{}, loss : {}'.format(i, loss.item()))
        torch.save(net.state_dict(), 'parameters.pkl')

    def validate(self, test_data, k):
        """
        compute the average NDCG@k for the given test data.
        :param test_data: test data
        :param k: used to compute NDCG@k
        :return:
        """
        # load model parameters
        net = Model(self.n_feature, self.h1_units, self.h2_units)
        net.load_state_dict(torch.load('parameters.pkl'))

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

if __name__ == '__main__':
    print('Load training data...')
    training_data = np.load('./dataset/train.npy')
    print('Load done.\n\n')

    model1 = RankNet(46, 512, 256, 10, 0.01)
    model1.fit(training_data)

    print('Validate...')
    test_data = np.load('./dataset/test.npy')
    model1.validate(test_data,10)


