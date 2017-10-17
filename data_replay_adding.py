from collections import Counter
from numpy import random

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
    num_train = 10 # must change latin square to change
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


edit_many_many_splits_for_target_word("rice")
edit_many_many_splits_for_target_word("immune")
edit_many_many_splits_for_target_word("borrow")
edit_many_many_splits_for_target_word("cowboys")
