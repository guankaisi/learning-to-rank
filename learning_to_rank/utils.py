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

def single_dcg(scores, i, j):
    """
    compute the single dcg that i-th element located j-th position
    :param scores:
    :param i:
    :param j:
    :return:
    """
    return (np.power(2, scores[i]) - 1) / np.log2(j+2)


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


def ndcg_k(scores, k):
    scores_k = scores[:k]
    dcg_k = dcg(scores_k)
    idcg_k = dcg(sorted(scores)[::-1][:k])
    if idcg_k == 0:
        return np.nan
    return dcg_k/idcg_k

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
    IDCG = idcg(true_scores)
    single_dcgs = {}
    for i, j in order_pairs:
        if (i, i) not in single_dcgs:
            single_dcgs[(i, i)] = single_dcg(true_scores, i, i)
        if (j, j) not in single_dcgs:
            single_dcgs[(j, j)] = single_dcg(true_scores, j, j)
        single_dcgs[(i, j)] = single_dcg(true_scores, i, j)
        single_dcgs[(j, i)] = single_dcg(true_scores, j, i)

    for i, j in order_pairs:
        delta = abs(single_dcgs[(i,j)] + single_dcgs[(j,i)] - single_dcgs[(i,i)] -single_dcgs[(j,j)])/IDCG
        rho = 1 / (1 + np.exp(temp_scores[i] - temp_scores[j]))
        lambdas[i] += rho * delta
        lambdas[j] -= rho * delta

        rho_complement = 1.0 - rho
        w[i] += rho * rho_complement * delta
        w[i] -= rho * rho_complement * delta

    return lambdas, w, qid

def split_pairs(order_pairs, true_scores):
    """
    split the pairs into two list, named relevant_doc and irrelevant_doc.
    relevant_doc[i] is prior to irrelevant_doc[i]

    :param order_pairs: ordered pairs of all queries
    :param ture_scores: scores of docs for each query
    :return: relevant_doc and irrelevant_doc
    """
    relevant_doc = []
    irrelevant_doc = []
    doc_idx_base = 0
    query_num = len(order_pairs)
    for i in range(query_num):
        pair_num = len(order_pairs[i])
        docs_num = len(true_scores[i])
        for j in range(pair_num):
            d1, d2 = order_pairs[i][j]
            d1 += doc_idx_base
            d2 += doc_idx_base
            relevant_doc.append(d1)
            irrelevant_doc.append(d2)
        doc_idx_base += docs_num
    return relevant_doc, irrelevant_doc

def data_reader(file_loc):
    f = open(file_loc, 'r')
    data = []
    for line in f:
        new_arr = []
        arr = line.split(' #')[0].split()
        score = arr[0]
        q_id = arr[1].split(':')[1]
        new_arr.append(int(score))
        new_arr.append(int(q_id))
        arr = arr[2:]
        for el in arr:
            new_arr.append(float(el.split(':')[1]))
        data.append(new_arr)
    f.close()
    return np.array(data)