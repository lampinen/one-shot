import numpy
import matplotlib.pyplot as plot

emb_filename = '../result_data/pre_embeddings/bonuses/embedding_pre_full.csv'
smb_filename = '../result_data/pre_embeddings/bonuses/softmax_b_pre_full.csv'
smw_filename = '../result_data/pre_embeddings/bonuses/softmax_w_pre_full.csv'

emb = numpy.genfromtxt(emb_filename, delimiter=',')

emb_centr = numpy.mean(emb, axis=0)
emb_centr_unit = emb_centr / numpy.sqrt(numpy.sum(numpy.square(emb_centr)))
emb_centr_projections = numpy.dot(emb, emb_centr_unit) / numpy.sqrt(numpy.sum(numpy.square(emb), axis=1))

emb_centr_dists = numpy.sqrt(numpy.sum(numpy.square(emb - emb_centr), axis=1))

approaches = ["centroid", "opt", "opt_centroid", "opt_zero"]
new_word_embeddings = {approach: numpy.zeros((100, 1500)) for approach in approaches} 

for word in ["bonuses"]:
    for num_train in xrange(1, 11):
	for perm in xrange(1, 11):
	    for approach in approaches:
		this_emb = numpy.genfromtxt("../result_data/%s_perm%i_train_logs/%s/embedding/num%i_%s_embedding_%s.csv" %(word, perm, word, num_train, approach, approach), delimiter=',')
		new_word_embeddings[approach][10*(num_train - 1) + perm -1, :] = this_emb


emb_lengths = numpy.sqrt(numpy.sum(numpy.square(emb), axis=1))
new_word_lengths = {approach: numpy.sqrt(numpy.sum(numpy.square(new_word_embeddings[approach]), axis=1)) for approach in approaches}

_, bins, _ = plot.hist(emb_lengths, bins=100, label='Other words')
for approach in approaches:
    plot.hist(new_word_lengths[approach], bins=bins, label=approach)

plot.legend(loc='upper right')
plot.title('l2 lengths of embeddings')
#plot.show()



#
#new_word_centr_dists = {approach: numpy.sqrt(numpy.sum(numpy.square(new_word_embeddings[approach] - emb_centr), axis=1)) for approach in approaches}
#
#_, bins, _ = plot.hist(emb_centr_dists, bins=100, label='Other words')
#for approach in approaches:
#    plot.hist(new_word_centr_dists[approach], bins=bins, label=approach)
#
#plot.legend(loc='upper right')
#plot.title('Euclidean distances from centroid')
#plot.show()
#
#withword_emb_filename = '../result_data/bonuses_withword_train_logs/bonuses/embedding/withword_embedding_opt.csv'
#withword_emb = numpy.genfromtxt(withword_emb_filename, delimiter=',')
#emb_ww_dists = numpy.sqrt(numpy.sum(numpy.square(emb - withword_emb), axis=1))
#	
#new_word_ww_dists = {approach: numpy.sqrt(numpy.sum(numpy.square(new_word_embeddings[approach] - withword_emb), axis=1)) for approach in approaches}
#
#_, bins, _ = plot.hist(emb_ww_dists, bins=100, label='Other words')
#for approach in approaches:
#    plot.hist(new_word_ww_dists[approach], bins=bins, label=approach)
#
#plot.legend(loc='upper right')
#plot.title('Euclidean distances from embedding when training with the words')
#plot.show()

emb_simils = numpy.zeros((100, 100))
for i in xrange(0, 100):
    for j in xrange(i, 100):
	emb_simils[i, j] = numpy.sqrt(numpy.sum(numpy.square(emb[i*100, :] - emb[j*100, :])))
	emb_simils[j, i] = emb_simils[i, j]
numpy.savetxt('intermediate_data/emb_other_simils.csv', emb_simils.flatten(), delimiter=',')

#plot.imshow(emb_simils)
#plot.show()


simils = {}

for i, a in enumerate(approaches):
    for b in approaches[i:]:
        this_simil = numpy.zeros((100, 100))
        for i in xrange(0, 100):
            for j in xrange(i, 100):
                this_simil[i, j] = numpy.sqrt(numpy.sum(numpy.square(new_word_embeddings[a][i, :] - new_word_embeddings[b][j, :])))
                this_simil[j, i] = this_simil[i, j]
        simils[(a, b)] = this_simil
	numpy.savetxt('intermediate_data/emb_%s_%s_simils.csv' % (a, b), this_simil.flatten(), delimiter=',')

colors = plot.cm.jet(numpy.linspace(0, 1, len(simils) + 1))
keys_and_values = simils.items()
labels = ["Other words"]
labels.extend([x[0] for x in keys_and_values])
simils_to_plot = [emb_simils.flatten()]
simils_to_plot.extend([x[1] for x in keys_and_values])
plot.hist(simils_to_plot, bins=100, label=labels, color=colors, histtype='barstacked')

plot.legend(loc='upper right')
plot.title('Distances between word embeddings')
#plot.show()




###### Softmax weights #######################
smw = numpy.genfromtxt(smw_filename, delimiter=',')
smw = numpy.transpose(smw) # Put into same orientation as emb

smw_centr = numpy.mean(smw, axis=0)
smw_centr_unit = smw_centr / numpy.sqrt(numpy.sum(numpy.square(smw_centr)))
smw_centr_projections = numpy.dot(smw, smw_centr_unit) / numpy.sqrt(numpy.sum(numpy.square(smw), axis=1))

smw_centr_dists = numpy.sqrt(numpy.sum(numpy.square(smw - smw_centr), axis=1))

approaches = ["centroid", "opt", "opt_centroid", "opt_zero"]
new_word_smws = {approach: numpy.zeros((100, 1500)) for approach in approaches} 

for word in ["bonuses"]:
    for num_train in xrange(1, 11):
	for perm in xrange(1, 11):
	    for approach in approaches:
		this_smw = numpy.genfromtxt("../result_data/%s_perm%i_train_logs/%s/embedding/num%i_%s_softmax_w_%s.csv" %(word, perm, word, num_train, approach, approach), delimiter=',')
		new_word_smws[approach][10*(num_train - 1) + perm -1, :] = this_smw



smw_simils = numpy.zeros((100, 100))
for i in xrange(0, 100):
    for j in xrange(i, 100):
	smw_simils[i, j] = numpy.sqrt(numpy.sum(numpy.square(smw[i*100, :] - smw[j*100, :])))
	smw_simils[j, i] = smw_simils[i, j]
numpy.savetxt('intermediate_data/smw_other_simils.csv', smw_simils.flatten(), delimiter=',')

#plot.imshow(smw_simils)
#plot.show()


simils = {}

for i, a in enumerate(approaches):
    for b in approaches[i:]:
        this_simil = numpy.zeros((100, 100))
        for i in xrange(0, 100):
            for j in xrange(i, 100):
                this_simil[i, j] = numpy.sqrt(numpy.sum(numpy.square(new_word_smws[a][i, :] - new_word_smws[b][j, :])))
                this_simil[j, i] = this_simil[i, j]
        simils[(a, b)] = this_simil
	numpy.savetxt('intermediate_data/smw_%s_%s_simils.csv' % (a, b), this_simil.flatten(), delimiter=',')

colors = plot.cm.jet(numpy.linspace(0, 1, len(simils) + 1))
keys_and_values = simils.items()
labels = ["Other words"]
labels.extend([x[0] for x in keys_and_values])
simils_to_plot = [smw_simils.flatten()]
simils_to_plot.extend([x[1] for x in keys_and_values])
plot.hist(simils_to_plot, bins=100, label=labels, color=colors, histtype='barstacked')

plot.legend(loc='upper right')
plot.title('Distances between word softmax weights')
#plot.show()
