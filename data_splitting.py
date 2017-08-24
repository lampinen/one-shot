from collections import Counter
from numpy import random

data_file = "raw_data/ptb.train.txt"
output_dir = "edited_data/"
output_file_prefix = "ptb.train."

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

word_counter = Counter()
with open(data_file, "r") as fin:
    for line in fin.readlines():
	word_counter.update(line.split())

#print(word_counter.most_common(500))

targets = [word for word in word_counter.keys() if word_counter[word] == 20]
print(targets)

def create_split_for_target_word(word, num_train=1):
    word_lines = []
    with open(data_file, "r") as fin:
	with open(output_dir + output_file_prefix + "no." + word + '.txt', "w") as noword_f:
		for line in fin.readlines():
		    if word in line.split():
			word_lines.append(line)
		    else:
			noword_f.write(line)

    with open(output_dir + output_file_prefix + word + '.wordtrain.txt', "w") as word_train_f:
	for i in xrange(num_train):
	    chosen_line = word_lines.pop(random.randint(len(word_lines)))
	    word_train_f.write(chosen_line)
    with open(output_dir + output_file_prefix + word + '.wordtest.txt', "w") as word_test_f:
	word_test_f.writelines(word_lines)
	
	

def create_many_splits_for_target_word(word, num_train=10):
    word_lines = []
    with open(data_file, "r") as fin:
	with open(output_dir + output_file_prefix + "no." + word + '.txt', "w") as noword_f:
		for line in fin.readlines():
		    if word in line.split():
			word_lines.append(line)
		    else:
			noword_f.write(line)

    random.shuffle(word_lines)
    assert(len(word_lines) > num_train)
    test_word_lines = word_lines[num_train:]
    train_word_lines = word_lines[:num_train]
    with open(output_dir + output_file_prefix + word + '.fixedwordtest.txt', "w") as word_test_f:
	word_test_f.writelines(test_word_lines)
    for i in xrange(1,num_train+1):
	with open(output_dir + output_file_prefix + word + '.' + str(i) + 'wordtrain.txt', "w") as word_train_f:
	    word_train_f.writelines(train_word_lines[:i])


def create_many_many_splits_for_target_word(word):
    num_train = 10 # must change latin square to change
    word_lines = []
    with open(data_file, "r") as fin:
	with open(output_dir + '/' + word + '_many_many/' + output_file_prefix + "no." + word + '.txt', "w") as noword_f:
	    for line in fin.readlines():
		if word in line.split():
		    word_lines.append(line)
		else:
		    noword_f.write(line)

    random.shuffle(word_lines)
    assert(len(word_lines) > num_train)
    test_word_lines = word_lines[num_train:]
    train_word_lines = word_lines[:num_train]

    with open(output_dir + '/' + word + '_many_many/' + output_file_prefix + word + '.fixedwordtest.txt', "w") as word_test_f:
	word_test_f.writelines(test_word_lines)
    for j in xrange(1,num_train+1):
	train_word_lines = train_word_lines
	for this_num in xrange(1,num_train+1):
	    with open(output_dir + '/' + word + '_many_many/' + output_file_prefix + word + 'perm' + str(j) + '.' + str(this_num) + 'wordtrain.txt', "w") as word_train_f:
		for k in xrange(this_num):
		    word_train_f.write(train_word_lines[latin_square_10[j-1][k-1]])

    with open(data_file, "r") as fin:
	with open(output_dir + '/' + word + '_many_many/' + output_file_prefix + "nofixedtest." + word + '.txt', "w") as noword_f:
	    for line in fin.readlines():
		if line not in test_word_lines:
		    noword_f.write(line)


#create_split_for_target_word("bonuses")

#create_many_splits_for_target_word("bonuses")
#create_many_splits_for_target_word("explained")
#create_many_splits_for_target_word("strategist")
#create_many_splits_for_target_word("entry")
#create_many_splits_for_target_word("rice")
#create_many_splits_for_target_word("marketers")
#create_many_splits_for_target_word("immune")
#create_many_splits_for_target_word("cowboys")
#create_many_splits_for_target_word("borrow")


create_many_many_splits_for_target_word("bonuses")
create_many_many_splits_for_target_word("explained")
create_many_many_splits_for_target_word("strategist")
