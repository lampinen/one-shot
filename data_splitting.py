from collections import Counter

data_file = "raw_data/ptb.train.txt"
output_dir = "edited_data/"
output_file_prefix = "ptb.train."


word_counter = Counter()
with open(data_file, "r") as fin:
    for line in fin.readlines():
	word_counter.update(line.split())

#print(word_counter.most_common(500))

targets = [word for word in word_counter.keys() if word_counter[word] == 20]
print(targets)

def create_split_for_target_word(word):

    with open(data_file, "r") as fin:
	with open(output_dir + output_file_prefix + "no." + word + '.txt', "w") as noword_f:
	    with open(output_dir + output_file_prefix + "only." + word + '.txt', "w") as word_f:
		for line in fin.readlines():
		    if word in line:
			word_f.write(line)
		    else:
			noword_f.write(line)

create_split_for_target_word("bonuses")
