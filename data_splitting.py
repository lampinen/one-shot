from collections import Counter
from numpy import random

data_file = "raw_data/ptb.train.txt"
output_dir = "edited_data/"
output_file_prefix = "ptb.train."
num_train = 1 #number of lines to use for training

word_counter = Counter()
with open(data_file, "r") as fin:
    for line in fin.readlines():
	word_counter.update(line.split())

#print(word_counter.most_common(500))

targets = [word for word in word_counter.keys() if word_counter[word] == 20]
print(targets)

def create_split_for_target_word(word):
    word_lines = []
    with open(data_file, "r") as fin:
	with open(output_dir + output_file_prefix + "no." + word + '.txt', "w") as noword_f:
		for line in fin.readlines():
		    if word in line:
			word_lines.append(line)
		    else:
			noword_f.write(line)

    with open(output_dir + output_file_prefix + word + '.wordtrain.txt', "w") as word_train_f:
	for i in xrange(num_train):
	    chosen_line = word_lines.pop(random.randint(len(word_lines)))
	    word_train_f.write(chosen_line)
    with open(output_dir + output_file_prefix + word + '.wordtest.txt', "w") as word_test_f:
	word_test_f.writelines(word_lines)
	
	

create_split_for_target_word("bonuses")
