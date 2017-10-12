import numpy
import matplotlib.pyplot as plot
import collections

approaches = ["opt_centroid", "centroid"]
words = ["immune", "cowboys", "rice", "borrow"]
vocabulary = numpy.loadtxt('../raw_data/vocabulary.csv', dtype=str)
vocabulary = numpy.array([word[:word.find(',')] for word in vocabulary])


#this_smw = smw[:, 48]
#inner_products = numpy.dot(this_smw, normalized_smw)
#top = inner_products.argsort()[-10:]
#print(vocabulary[top])
#this_smw = smw[:, 48]
#inner_products = numpy.dot(this_smw, smw)
#top = inner_products.argsort()[-10:]
#print(vocabulary[top])
#exit()



top_words = {word: {approach: collections.Counter() for approach in approaches} for word in words}
for word in words:
    smw_filename = '../result_data_4/pre_embeddings/%s/softmax_w_pre_full.csv' % word
    smw = numpy.genfromtxt(smw_filename, delimiter=',')
    normalized_smw = smw / numpy.sqrt(numpy.sum(smw**2, axis=0))

    for num_train in xrange(1, 11):
	for perm in xrange(1, 11):
	    for approach in approaches:
		this_smw = numpy.genfromtxt("../result_data_4/%s_perm%i_train_logs/%s/embedding/num%i_%s_softmax_w_%s.csv" %(word, perm, word, num_train, approach, approach), delimiter=',')
		inner_products = numpy.dot(this_smw, smw)
		top = inner_products.argsort()[-10:]
		top_words[word][approach].update(vocabulary[top])

    print word 
    print top_words[word]["centroid"]
    print top_words[word]["opt_centroid"]
