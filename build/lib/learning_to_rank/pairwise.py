import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostRegressor
from learning_to_rank.utils import group_by, get_pairs, compute_lambda, ndcg_k, split_pairs


class RankNet_Net(nn.Module):
    """
    construct the RankNet
    """
    def __init__(self, n_feature, h1_units, h2_units):
        super(RankNet_Net, self).__init__()

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
        self.model = RankNet_Net(n_feature, h1_units, h2_units)
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
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
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
        torch.save(net.state_dict(), 'ranknet.pkl')

    def validate(self, test_data, k):
        """
        compute the average NDCG@k for the given test data.
        :param test_data: test data
        :param k: used to compute NDCG@k
        :return:
        """
        # load model parameters
        net = self.model(self.n_feature, self.h1_units, self.h2_units)
        net.load_state_dict(torch.load('ranknet.pkl'))

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


class LambdaRank_Net(nn.Module):
    def __init__(self, n_feature, h1_units, h2_units):
        super(LambdaRank_Net, self).__init__()
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


class LambdaRank:

    def __init__(self, training_data, n_feature, h1_units = 512, h2_units = 256, epoch = 10, lr=0.001):
        self.training_data = training_data
        self.n_feature = n_feature
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.epoch = epoch
        self.lr = lr
        self.trees = []
        self.model = LambdaRank_Net(n_feature, h1_units, h2_units)
        # for para in self.model.parameters():
        #     print(para[0])

    def fit(self):
        """
        train the model to fit the train dataset
        """
        qid_doc_map = group_by(self.training_data, 1)
        query_idx = qid_doc_map.keys()
        # true_scores is a matrix, different rows represent different queries
        true_scores = [self.training_data[qid_doc_map[qid], 0] for qid in query_idx]

        order_paris = []
        for scores in true_scores:
            order_paris.append(get_pairs(scores))

        sample_num = len(self.training_data)
        print('Training .....')
        for i in range(self.epoch):
            predicted_scores = self.model(torch.from_numpy(self.training_data[:, 2:].astype(np.float32)))
            predicted_scores_numpy = predicted_scores.data.numpy()
            lambdas = np.zeros(sample_num)
            # w = np.zeros(sample_num)

            pred_score = [predicted_scores_numpy[qid_doc_map[qid]] for qid in query_idx]

            zip_parameters = zip(true_scores, pred_score, order_paris, query_idx)
            for ts, ps, op, qi in zip_parameters:
                sub_lambda, sub_w, qid = compute_lambda(ts, ps, op, qi)
                lambdas[qid_doc_map[qid]] = sub_lambda
                # w[qid_doc_map[qid]] = sub_w
            # update parameters
            self.model.zero_grad()
            lambdas_torch = torch.Tensor(lambdas).view((len(lambdas), 1))
            predicted_scores.backward(lambdas_torch, retain_graph=True)
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data.add_(param.grad.data * self.lr)


            if i % 1 == 0:
                qid_doc_map = group_by(self.training_data, 1)
                ndcg_list = []
                for qid in qid_doc_map.keys():
                    subset = qid_doc_map[qid]

                    X_subset = torch.from_numpy(self.training_data[subset, 2:].astype(np.float32))
                    sub_pred_score = self.model(X_subset).data.numpy().reshape(1, len(X_subset)).squeeze()

                    # calculate the predicted NDCG
                    true_label = self.training_data[qid_doc_map[qid], 0]
                    k = len(true_label)
                    pred_sort_index = np.argsort(sub_pred_score)[::-1]
                    true_label = true_label[pred_sort_index]
                    ndcg_val = ndcg_k(true_label, k)
                    ndcg_list.append(ndcg_val)
                print('Epoch:{}, Average NDCG : {}'.format(i, np.nanmean(ndcg_list)))


    def predict(self, data):
        """
        predict the score for each document in testset
        :param data: given testset
        :return:
        """
        qid_doc_map = group_by(data, 1)
        predicted_scores = np.zeros(len(data))
        for qid in qid_doc_map.keys():
            subset = qid_doc_map[qid]
            X_subset = torch.from_numpy(data[subset, 2:].astype(np.float32))
            sub_pred_score = self.model(X_subset).data.numpy().reshape(1, len(X_subset)).squeeze()
            predicted_scores[qid_doc_map[qid]] = sub_pred_score
        return predicted_scores

    def validate(self, data, k):
        """
        validate the NDCG metric
        :param data: given th testset
        :param k: used to compute the NDCG@k
        :return:
        """
        qid_doc_map = group_by(data, 1)
        ndcg_list = []
        predicted_scores = np.zeros(len(data))
        for qid in qid_doc_map.keys():
            subset = qid_doc_map[qid]
            X_subset = torch.from_numpy(data[subset, 2:].astype(np.float32))
            sub_pred_score = self.model(X_subset).data.numpy().reshape(1, len(X_subset)).squeeze()

            # calculate the predicted NDCG
            true_label = data[qid_doc_map[qid], 0]

            pred_sort_index = np.argsort(sub_pred_score)[::-1]
            true_label = true_label[pred_sort_index]
            ndcg_val = ndcg_k(true_label, k)
            ndcg_list.append(ndcg_val)
        average_ndcg = np.nanmean(ndcg_list)
        return average_ndcg, predicted_scores

class RankSVM():
    """
    user interface
    """
    def __init__(self, training_data ,kernel='linear',max_iter=-1):
        self.training_data = training_data

        self.kernel = 'linear'
        self.max_iter = -1
        self.model = SVC(kernel=self.kernel,max_iter=self.max_iter)

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

        X1 = relevant_doc[:, 2:][:5000]
        X2 = irrelevant_doc[:, 2:][:5000]

        X = np.concatenate((X1,X2),axis=0)

        y1 = np.ones((X1.shape[0]))
        y2 = np.zeros((X2.shape[0]))

        Y = np.concatenate((y1,y2),axis=0)
        # training......
        self.model.fit(X,Y)


    def validate(self, data, k):
        """
        validate the NDCG metric
        :param data: given th testset
        :param k: used to compute the NDCG@k
        :return:
        """
        qid_doc_map = group_by(data, 1)
        ndcg_list = []
        predicted_scores = np.zeros(len(data))
        for qid in qid_doc_map.keys():
            subset = qid_doc_map[qid]
            X_subset = torch.from_numpy(data[subset, 2:].astype(np.float32))
            sub_pred_score = self.model.predict(X_subset).reshape(1, len(X_subset)).squeeze()
            # calculate the predicted NDCG
            true_label = data[qid_doc_map[qid], 0]

            pred_sort_index = np.argsort(sub_pred_score)[::-1]
            true_label = true_label[pred_sort_index]
            ndcg_val = ndcg_k(true_label, k)
            ndcg_list.append(ndcg_val)
        average_ndcg = np.nanmean(ndcg_list)
        return average_ndcg, predicted_scores


class RankBoost():
    """
    user interface
    """
    def __init__(self, training_data,random_state=0,n_estimators=100):
        self.training_data = training_data
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model = AdaBoostRegressor(random_state=self.random_state,n_estimators=self.n_estimators)
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

        X = np.concatenate((X1,X2),axis=0)

        y1 = np.ones((X1.shape[0]))
        y2 = np.zeros((X2.shape[0]))

        Y = np.concatenate((y1,y2),axis=0)
        # training......
        self.model.fit(X,Y)


    def validate(self, data, k):
        """
        validate the NDCG metric
        :param data: given th testset
        :param k: used to compute the NDCG@k
        :return:
        """
        qid_doc_map = group_by(data, 1)
        ndcg_list = []
        predicted_scores = np.zeros(len(data))
        for qid in qid_doc_map.keys():
            subset = qid_doc_map[qid]
            X_subset = torch.from_numpy(data[subset, 2:].astype(np.float32))
            sub_pred_score = self.model.predict(X_subset).reshape(1, len(X_subset)).squeeze()
            # calculate the predicted NDCG
            true_label = data[qid_doc_map[qid], 0]

            pred_sort_index = np.argsort(sub_pred_score)[::-1]
            true_label = true_label[pred_sort_index]
            ndcg_val = ndcg_k(true_label, k)
            ndcg_list.append(ndcg_val)
        average_ndcg = np.nanmean(ndcg_list)
        return average_ndcg, predicted_scores