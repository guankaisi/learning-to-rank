from  learning_to_rank.listwise import LambdaMART,ListNet,ListMLE
from learning_to_rank.pairwise import RankNet,LambdaRank,RankSVM,RankBoost
from learning_to_rank.poinstwise import NRegression
from learning_to_rank.utils import data_reader
import numpy as np
from sklearn import tree



def main():
	total_ndcg = 0.0
	for i in [1,2,3,4,5]:
		print('start Fold ' + str(i))
		training_data = data_reader('MQ2008/Fold%d/train.txt' % (i))
		test_data = data_reader('MQ2008/Fold%d/test.txt' % (i))
		model = LambdaMART(training_data, number_of_trees=500, lr=0.03,max_depth=4)
		# model = RankBoost(training_data)
		model.fit()
		# import matplotlib.pyplot as plt
		# tree.plot_tree(model.trees[0])
		# plt.show()
		average_ndcg, predicted_scores = model.validate(test_data, 10)
		print(average_ndcg)
		total_ndcg += average_ndcg
	total_ndcg /= 5.0
	print('average ndcg at 10 is: ' + str(total_ndcg))

	# model = RankNet(training_data, n_feature=46)
	# model = LambdaRank(training_data, n_feature)

	# model.save('lambdamart_model_%d' % (i))
	# model = LambdaMART()
	# model.load('lambdamart_model.lmart')

if __name__ == '__main__':
	main()