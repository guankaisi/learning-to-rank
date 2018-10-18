import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dcg(scores):
    """
    compute the DCG value based on the given score
    :param scores: a score list of documents
    :return v: DCG value
    """
    v = 0
    for i in range(len(scores)):
        v += (np.power(2, scores[i]) - 1) / np.log2(i+2)  # i+2 is because i starts from 0
    return v


def idcg(scores):
    """
    compute the IDCG value (best dcg value) based on the given score
    :param scores: a score list of documents
    :return:  IDCG value
    """
    best_scores = sorted(scores)[::-1]
    return dcg(best_scores)


def ndcg(scores):
    """
    compute the NDCG value based on the given score
    :param scores: a score list of documents
    :return:  NDCG value
    """
    return dcg(scores)/idcg(scores)


def delta_ndcg(scores, p, q):
    """
    swap the i-th and j-th doucment, compute the absolute value of NDCG delta
    :param scores: a score list of documents
    :param p, q: the swap positions of documents
    :return: the absolute value of NDCG delta
    """
    s2 = scores.copy()  # new score list
    s2[p], s2[q] = s2[q], s2[p]  # swap
    return abs(ndcg(s2) - ndcg(scores))


def ndcg_k(scores, k):
    scores_k = scores[:k]
    fenzi = dcg(scores_k)
    fenmu = dcg(sorted(scores)[::-1][:k])
    return fenzi/fenmu

def group_by(data, qid_index):
    """

    :param data: input_data
    :param qid_index: the column num where qid locates in input data
    :return: a dict group by qid
    """
    qid_doc_map = {}
    idx = 0
    for record in data:
        qid_doc_map.setdefault(record[qid_index], [])
        qid_doc_map[record[qid_index]].append(idx)
        idx += 1
    return qid_doc_map


def get_pairs(scores):
    """

    :param scores: given score list of documents for a particular query
    :return: the documents pairs whose firth doc has a higher value than second one.
    """
    pairs = []
    for i in range(len(scores)):
        for j in range(len(scores)):
            if scores[i] > scores[j]:
                pairs.append((i, j))
    return pairs


def compute_lambda(true_scores, temp_scores, order_pairs, qid):
    """

    :param true_scores: the score list of the documents for the qid query
    :param temp_scores: the predict score list of the these documents
    :param order_pairs: the partial oder pairs where first document has higher score than the second one
    :param qid: specific query id
    :return:
        lambdas: changed lambda value for these documents
        w: w value
        qid: query id
    """
    doc_num = len(true_scores)
    lambdas = np.zeros(doc_num)
    w = np.zeros(doc_num)
    for i, j in order_pairs:
        delta = delta_ndcg(true_scores, i, j)
        rho = 1 / (1 + np.exp(temp_scores[i] - temp_scores[j]))
        lambdas[i] += rho * delta
        lambdas[j] -= rho * delta

        rho_complement = 1.0 - rho
        w[i] += rho * rho_complement * delta
        w[i] -= rho * rho_complement * delta

    return lambdas, w, qid


def load_data(file_path='/Users/hou/OneDrive/KDD2019/data/L2R/sample.txt'):
    with open(file_path, 'r') as f:
        data = []
        for line in f.readlines():
            new_arr = []
            line_split = line.split(' ')
            score = float(line_split[0])
            qid = int(line_split[1].split(':')[1])
            new_arr.append(score)
            new_arr.append(qid)
            for ele in line_split[2:]:
                new_arr.append(float(ele.split(':')[1]))
            data.append(new_arr)
    data_np = np.array(data)
    return data_np


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


class LambdaRank:

    def __init__(self, training_data, n_feature, h1_units, h2_units, epoch, lr=0.001):
        self.training_data = training_data
        self.n_feature = n_feature
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.epoch = epoch
        self.lr = lr
        self.trees = []
        self.model = Net(n_feature, h1_units, h2_units)

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
        print('Training .....\n')
        for i in range(self.epoch):
            predicted_scores = self.model(torch.from_numpy(self.training_data[:, 2:].astype(np.float32)))
            predicted_scores_numpy = predicted_scores.data.numpy()
            lambdas = np.zeros(sample_num)
            w = np.zeros(sample_num)

            pred_score = [predicted_scores_numpy[qid_doc_map[qid]] for qid in query_idx]

            zip_parameters = zip(true_scores, pred_score, order_paris, query_idx)
            count = 0
            for ts, ps, op, qi in zip_parameters:
                print(count)
                count+=1
                sub_lambda, sub_w, qid = compute_lambda(ts, ps, op, qi)
                lambdas[qid_doc_map[qid]] = sub_lambda
                w[qid_doc_map[qid]] = sub_w
            print('%d times update.....' %i)
            # update parameters
            self.model.zero_grad()
            lambdas = torch.Tensor(lambdas).view((len(lambdas), 1))
            predicted_scores.backward(lambdas)  # This is very important. Please understand why?
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data.add_(param.grad.data * self.lr)


            if i % 10 == 0:
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
                print('Epoch:{}, Average NDCG : {}'.format(i, np.mean(ndcg_list)))


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
        return ndcg_list


if __name__ == '__main__':
    # data = load_data()
    training_data = np.load('train data path')
    n_feature = training_data.shape[1] - 2
    h1_units = 128
    h2_units = 64
    epoch = 100
    learning_rate = 0.01
    model = LambdaRank(training_data, n_feature, h1_units, h2_units, epoch, learning_rate)
    model.fit()
    k = 4
    test_data = np.load('test data path')
    ndcg = model.validate(test_data, k)
    print(ndcg)