import numpy
from scipy.spatial.distance import pdist, cdist
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



top_words = {word: {approach: collections.Counter() for approach in approaches + ["with"]} for word in words}


word_similarities = {otherword: {newword: None for newword in words if otherword != newword} for otherword in words} 
otherword_sim_corrs = {word: [] for word in words}
newword_sim_corrs = {word: {num_train: []  for num_train in xrange(1,11)} for word in words}


for word in words:
    print "Loading %s" % word
    smw_filename = '../result_data_4/pre_embeddings/%s/softmax_w_pre_full.csv' % word
    smw = numpy.genfromtxt(smw_filename, delimiter=',')
#    smw =smw / numpy.linalg.norm(smw, axis=0) 
    for otherword in words:
	if otherword == word:
	    continue
	index = numpy.argwhere(vocabulary == otherword)[0]
	inner_products = numpy.dot(numpy.transpose(smw[:, index]), smw)[0] 
	word_similarities[otherword][word] = numpy.delete(inner_products, index) 

with open('../result_data_4/simil_corrs.csv', 'w') as fout:
    fout.write('word, comp_number, approach, perm, num_train, similarity_correlation\n')
    for word in words:
	otherword_sim_corrs[word] = pdist(numpy.array(word_similarities[word].values()), 'correlation')
	for i in range(3):
	    fout.write('%s, %i, %s, %i, %i, %f\n' % (word, i, 'with', 1, 10, otherword_sim_corrs[word][i]))

    for word in words:
	smw_filename = '../result_data_4/pre_embeddings/%s/softmax_w_pre_full.csv' % word
	smw = numpy.genfromtxt(smw_filename, delimiter=',')
     #   smw = smw / numpy.linalg.norm(smw, axis=0) 

	for approach in approaches:
	    for num_train in xrange(1, 11):
		for perm in xrange(1, 11):
			this_smw = numpy.genfromtxt("../result_data_4/%s_perm%i_train_logs/%s/embedding/num%i_%s_softmax_w_%s.csv" %(word, perm, word, num_train, approach, approach), delimiter=',')
    #		    this_smw = this_smw / numpy.linalg.norm(this_smw)
			inner_products = numpy.dot(this_smw, smw)
			index = numpy.argwhere(vocabulary == otherword)[0]
			sim_corrs = cdist(numpy.array(word_similarities[word].values()), [numpy.delete(inner_products, index)], 'correlation')
			newword_sim_corrs[word][num_train].append(sim_corrs)
			for i in range(3):
			    fout.write('%s, %i, %s, %i, %i, %f\n' % (word, i, approach, perm, num_train, sim_corrs[i]))
		print word, approach, num_train, numpy.mean(newword_sim_corrs[word][num_train])

    #		    top = inner_products.argsort()[-10:]
    #		    top_words[word][approach].update(vocabulary[top])
    #	    print word, approach, num_train, numpy.mean(norms)


	

    #    print word 
    #    print top_words[word]["with"]
    #    print top_words[word]["opt_centroid"]
