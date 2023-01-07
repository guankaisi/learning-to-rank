from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from learning_to_rank.utils import group_by, get_pairs, compute_lambda, ndcg_k


class LambdaMART:
    def __init__(self, training_data=None, number_of_trees=10, lr = 0.001,max_depth=50):
        self.training_data = training_data
        self.number_of_trees = number_of_trees
        self.lr = lr
        self.trees = []
        self.max_depth = max_depth

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
        predicted_scores = np.zeros(sample_num)
        for k in range(self.number_of_trees):
            print('Tree %d' % k)
            lambdas = np.zeros(sample_num)
            w = np.zeros(sample_num)
            temp_score = [predicted_scores[qid_doc_map[qid]] for qid in query_idx]
            zip_parameters = zip(true_scores, temp_score, order_paris, query_idx)

            for ts, temps, op, qi in zip_parameters:
                sub_lambda, sub_w, qid = compute_lambda(ts, temps, op, qi)
                lambdas[qid_doc_map[qid]] = sub_lambda
                w[qid_doc_map[qid]] = sub_w
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(self.training_data[:, 2:], lambdas)
            self.trees.append(tree)
            pred = tree.predict(self.training_data[:, 2:])
            predicted_scores += self.lr * pred

            # print NDCG
            qid_doc_map = group_by(self.training_data, 1)
            ndcg_list = []
            for qid in qid_doc_map.keys():
                subset = qid_doc_map[qid]
                sub_pred_score = predicted_scores[subset]

                # calculate the predicted NDCG
                true_label = self.training_data[qid_doc_map[qid], 0]
                topk = len(true_label)
                pred_sort_index = np.argsort(sub_pred_score)[::-1]
                true_label = true_label[pred_sort_index]
                ndcg_val = ndcg_k(true_label, topk)
                ndcg_list.append(ndcg_val)
            # print('Epoch:{}, train dataset: NDCG : {}'.format(k, np.nanmean(ndcg_list)))

    def predict(self, data):
        """
        predict the score for each document in testset
        :param data: given testset
        :return:
        """
        qid_doc_map = group_by(data, 1)
        predicted_scores = np.zeros(len(data))
        for qid in qid_doc_map.keys():
            sub_result = np.zeros(len(qid_doc_map[qid]))
            for tree in self.trees:
                sub_result += self.lr * tree.predict(data[qid_doc_map[qid], 2:])
            predicted_scores[qid_doc_map[qid]] = sub_result
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
            sub_pred_result = np.zeros(len(qid_doc_map[qid]))
            for tree in self.trees:
                sub_pred_result += self.lr * tree.predict(data[qid_doc_map[qid], 2:])
            predicted_scores[qid_doc_map[qid]] = sub_pred_result
            # calculate the predicted NDCG
            true_label = data[qid_doc_map[qid], 0]
            pred_sort_index = np.argsort(sub_pred_result)[::-1]
            true_label = true_label[pred_sort_index]
            ndcg_val = ndcg_k(true_label, k)
            ndcg_list.append(ndcg_val)
        average_ndcg = np.nanmean(ndcg_list)
        return average_ndcg, predicted_scores

    def save(self, fname):
        pickle.dump(self, open('%s.lmart' % (fname), "wb"), protocol=2)

    def load(self, fname):
        model = pickle.load(open(fname, "rb"))
        self.training_data = model.training_data
        self.number_of_trees = model.number_of_trees
        self.learning_rate = model.learning_rate
        self.trees = model.trees

class ListNet_Net(nn.Module):
    def __init__(self,n_feature,h1_units,h2_units):
        super(ListNet_Net,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_feature,h1_units),
            nn.ReLU(),
            nn.Linear(h1_units,h2_units),
            nn.ReLU(),
            nn.Linear(h2_units,1)
        )
    def forward(self,x):
        x = self.model(x)
        return x

    def predict(self,x):
        x = self.model(x)
        return x.data.numpy()[0]

class ListNet():
    """
    user interface
    """
    def __init__(self, training_data ,n_feature, h1_units = 512, h2_units = 256, epoch = 10, learning_rate = 0.01):
        self.training_data = training_data
        self.n_feature = n_feature
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.model = ListNet_Net(n_feature, h1_units, h2_units)
        self.epoch = epoch
        self.learning_rate = learning_rate

    def decay_learning_rate(self, optimizer, epoch, decay_rate):
        if (epoch+1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay_rate

    #listnet的损失函数实际上就是一个交叉熵函数
    def listnet_loss(self,y_pred,y):
        y = F.softmax(y,dim=0)
        y_pred = F.softmax(y_pred,dim=0)
        return -torch.sum(y * torch.log(y_pred))



    def fit(self):
        net = self.model
        qid_doc_map = group_by(self.training_data, 1)
        query_idx = qid_doc_map.keys()
        # true_scores is a matrix, different rows represent different queries
        tmp = [self.training_data[qid_doc_map[qid], 0] for qid in query_idx]
        true_scores = []
        for i in tmp:
            true_scores += (list(i))
        true_scores = torch.Tensor(true_scores).reshape(-1, 1)

        sample_num = len(self.training_data)
        loss_fn = self.listnet_loss
        loss_list = []
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        print('Training .....')
        for i in range(self.epoch):
            predicted_scores = self.model(torch.from_numpy(self.training_data[:, 2:].astype(np.float32)))

            self.decay_learning_rate(optimizer, i, 0.95)
            net.zero_grad()
            loss = loss_fn(predicted_scores, true_scores)
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
        net = ListNet_Net(self.n_feature, self.h1_units, self.h2_units)
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

class ListMLE_Net(nn.Module):
    def __init__(self,n_feature,h1_units,h2_units):
        super(ListMLE_Net,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_feature,h1_units),
            nn.ReLU(),
            nn.Linear(h1_units,h2_units),
            nn.ReLU(),
            nn.Linear(h2_units,1)
        )
    def forward(self,x):
        x = self.model(x)
        return x

    def predict(self,x):
        x = self.model(x)
        return x.data.numpy()[0]

class ListMLE():
    """
    user interface
    """
    def __init__(self, training_data ,n_feature, h1_units = 512, h2_units = 256, epoch = 10, learning_rate = 0.01):
        self.training_data = training_data
        self.n_feature = n_feature
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.model = ListNet_Net(n_feature, h1_units, h2_units)
        self.epoch = epoch
        self.learning_rate = learning_rate

    def decay_learning_rate(self, optimizer, epoch, decay_rate):
        if (epoch+1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay_rate

    # 这是核心部分，是listnet和listmle的差别所在
    def listmle_loss(self,y_pred,y):
        index = torch.argsort(input=y.reshape(-1), descending=True)
        y_pred = y_pred.reshape(-1)[index]
        y_tmp = torch.zeros(y_pred.shape)
        num = len(y_pred)

        for i in range(num):
            y_tmp[i] = -torch.log(F.softmax(y_pred[i:],dim=0)[0])
        loss = torch.sum(y_tmp)
        return loss
    def fit(self):
        net = self.model
        qid_doc_map = group_by(self.training_data, 1)
        query_idx = qid_doc_map.keys()
        # true_scores is a matrix, different rows represent different queries
        tmp = [self.training_data[qid_doc_map[qid], 0] for qid in query_idx]
        true_scores = []
        for i in tmp:
            true_scores += (list(i))
        true_scores = torch.Tensor(true_scores).reshape(-1, 1)

        sample_num = len(self.training_data)
        loss_fn = self.listmle_loss
        loss_list = []
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        print('Training .....')
        for i in range(self.epoch):
            predicted_scores = self.model(torch.from_numpy(self.training_data[:, 2:].astype(np.float32)))

            self.decay_learning_rate(optimizer, i, 0.95)
            net.zero_grad()
            loss = loss_fn(predicted_scores, true_scores)
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
        net = ListNet_Net(self.n_feature, self.h1_units, self.h2_units)
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


