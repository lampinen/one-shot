from collections import Counter
from numpy import random, argwhere, log

data_file = "raw_data/ptb.train.txt"
output_dir = "edited_data"
output_file_prefix = "ptb.train."
num_replay_lines = 1000

latin_square_10 = [[0, 1, 9, 2, 8, 3, 7, 4, 6, 5],
		   [1, 2, 0, 3, 9, 4, 8, 5, 7, 6],
		   [2, 3, 1, 4, 0, 5, 9, 6, 8, 7],
		   [3, 4, 2, 5, 1, 6, 0, 7, 9, 8],
		   [4, 5, 3, 6, 2, 7, 1, 8, 0, 9],
		   [5, 6, 4, 7, 3, 8, 2, 9, 1, 0],
		   [6, 7, 5, 8, 4, 9, 3, 0, 2, 1],
		   [7, 8, 6, 9, 5, 0, 4, 1, 3, 2],
		   [8, 9, 7, 0, 6, 1, 5, 2, 4, 3],
		   [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]]


replay_line_numbers = random.randint(42048, size=num_replay_lines)

def edit_many_many_splits_for_target_word(word):
    num_train = 10
    replay_lines = []
    with open(output_dir + '/' + word + '_many_many/' + output_file_prefix + "no." + word + '.txt', "r") as fin:
	for i, line in enumerate(fin.readlines()):
	    if i in replay_line_numbers:
		replay_lines.append(line)

    random.shuffle(replay_lines)

    for j in xrange(1,num_train+1):
	for this_num in xrange(1,num_train+1):
	    with open(output_dir + '/' + word + '_many_many/' + output_file_prefix + word + 'perm' + str(j) + '.' + str(this_num) + 'wordtrain.txt', "r") as word_train_f:
		curr_lines = word_train_f.readlines()
	    new_lines = curr_lines + replay_lines[:100*this_num]
	    random.shuffle(new_lines)
	    with open(output_dir + '/' + word + '_many_many/' + output_file_prefix + word + 'perm' + str(j) + '.' + str(this_num) + 'wordtrain.txt', "w") as word_train_f:
		word_train_f.writelines(new_lines)

def word_overlap_metric(x, y):
    y_split = y.split()
    x_split = x.split()
    return sum([a in y_split for a in x.split() if a != "<unk>"]) / (0.5 * (len(x_split) + len(y_split)))


word_counts = Counter()
with open('raw_data/ptb.train.txt', 'r') as ftrain:
    for line in ftrain.readlines():
                word_counts.update(line.split())

total_count = sum(word_counts.values())
log_probabilities = dict(word_counts)
for k, v in log_probabilities.items():
        log_probabilities[k] = float(v) / total_count

def smarter_word_overlap_metric(x, y):
    y_split = y.split()
    x_split = x.split()
    return sum([-log_probabilities[a] for a in x_split if a in y_split and a != "<unk>"])

def edit_many_many_splits_for_target_word_SW(word, metric=smarter_word_overlap_metric):
    num_train = 10

    for j in xrange(1,num_train+1):
	for this_num in xrange(1,num_train+1):
	    with open(output_dir + '/' + word + '_many_many/' + output_file_prefix + word + 'perm' + str(j) + '.' + str(this_num) + 'wordtrain.txt', "r") as word_train_f:
		curr_lines = word_train_f.readlines()
            curr_lines = filter(lambda x: word in x.split(), curr_lines) # filter old replay stuff if any out

            replay_lines = []
            replay_metrics = []
            with open(output_dir + '/' + word + '_many_many/' + output_file_prefix + "no." + word + '.txt', "r") as fin:
                for line in fin.readlines():
                    this_metric = sum(metric(line, this_line) for this_line in curr_lines)
                    if len(replay_lines) < 100 * this_num or this_metric > replay_metrics[-1]:
                        x = argwhere(this_metric > replay_metrics) 
                        if x != []:
                            replay_lines.insert(x[0], line)
                            replay_metrics.insert(x[0], this_metric) 
                            replay_lines.pop(-1)
                            replay_metrics.pop(-1)
                        else:
                            replay_lines.append(line)
                            replay_metrics.append(this_metric) 
            random.shuffle(replay_lines)

	    new_lines = curr_lines + replay_lines
	    random.shuffle(new_lines)
	    with open(output_dir + '/' + word + '_many_many/' + output_file_prefix + word + 'perm' + str(j) + '.' + str(this_num) + 'SWwordtrain.txt', "w") as word_train_f:
		word_train_f.writelines(new_lines)

edit_many_many_splits_for_target_word_SW("rice")
edit_many_many_splits_for_target_word_SW("immune")
edit_many_many_splits_for_target_word_SW("borrow")
edit_many_many_splits_for_target_word_SW("cowboys")

hundred_words = ['ab', 'absolutely', 'agricultural', 'aim', 'animals', 'announcing', 'arguments', 'assist', 'averaged', 'bass', 'bullish', 'calculations', 'carefully', 'claiming', 'compare', 'conceded', 'congressman', 'consortium', 'contest', 'creation', 'cumulative', 'danger', 'darman', 'die', 'discrimination', 'disney', 'dominant', 'dorrance', 'edwards', 'efficiency', 'elderly', 'enable', 'encouraging', 'entry', 'environmentalists', 'execution', 'expenditures', 'facts', 'formula', 'gaf', 'geneva', 'globe', 'golf', 'healthcare', 'homeless', 'honor', 'horse', 'incest', 'informed', 'investigators', 'iron', 'jackson', 'judgment', 'knight', 'lake', 'lend', 'louisville', 'lowest', 'lucrative', 'maturing', 'minute', 'mississippi', 'motorola', 'museum', 'nabisco', 'netherlands', 'nigel', 'nine-month', 'owning', 'petrochemical', 'pioneer', 'prepare', 'print', 'pro-choice', 'recognized', 'referred', 'regarded', 'rejection', 'requests', 'resorts', 'responsibilities', 'rolled', 'sansui', 'serving', 'setback', 'similarly', 'somewhere', 'sounds', 'staffers', 'stolen', 'treasurys', 'treat', 'truth', 'utah', 'vulnerable', 'ward', 'warsaw', 'wedtech', 'wheat', 'wisconsin'] 

replay_line_numbers = random.randint(40175, size=num_replay_lines)

def edit_files_for_hundred_words():
    replay_lines = []
    with open(output_dir + '/hundred_words/' + output_file_prefix + "no.hundredwords.txt", "r") as fin:
        for i, line in enumerate(fin.readlines()):
            if i in replay_line_numbers:
                replay_lines.append(line)

    random.shuffle(replay_lines)
    for word in hundred_words:
        with open(output_dir + '/hundred_words/' + output_file_prefix + word + '.wordtrain.txt', "r") as word_train_f:
            curr_lines = word_train_f.readlines()
        new_lines = curr_lines + replay_lines
        random.shuffle(new_lines)
        with open(output_dir + '/hundred_words/' + output_file_prefix + word + '.wordtrain.txt', "w") as word_train_f:
            word_train_f.writelines(new_lines)
                
#edit_files_for_hundred_words()
