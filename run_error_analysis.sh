#!/bin/bash

for word in borrow cowboys immune rice
do
    for perm in 1
    do
	for numtrain in 1 10
	do
	    for approach in centroid opt_centroid
	    do
		embedding_location=result_data_6/${word}_perm${perm}_train_logs/${word}/embedding
		test_file=edited_data/${word}_many_many/ptb.train.${word}.fixedwordtest.txt
		python ptb_error_analysis.py --train_file_path=edited_data/${word}_many_many/ptb.train.no.${word}.txt --word_test_file_path=${test_file} --new_word=${word} --model_save_path=${word}_train_logs/${word}/pre_fine/ --output_prefix=perm${perm}_numtrain${numtrain}_wordpresent_ --reload_embedding_path=${embedding_location}/num${numtrain}_${approach}_embedding_${approach}.csv --reload_softmax_b_path=${embedding_location}/num${numtrain}_${approach}_softmax_b_${approach}.csv --reload_softmax_w_path=${embedding_location}/num${numtrain}_${approach}_softmax_w_${approach}.csv
		test_file=edited_data/error_falsepositive_test.txt
		python ptb_error_analysis.py --train_file_path=edited_data/${word}_many_many/ptb.train.no.${word}.txt --word_test_file_path=${test_file} --new_word=${word} --model_save_path=${word}_train_logs/${word}/pre_fine/ --output_prefix=perm${perm}_numtrain${numtrain}_wordabsent_ --reload_embedding_path=${embedding_location}/num${numtrain}_${approach}_embedding_${approach}.csv --reload_softmax_b_path=${embedding_location}/num${numtrain}_${approach}_softmax_b_${approach}.csv --reload_softmax_w_path=${embedding_location}/num${numtrain}_${approach}_softmax_w_${approach}.csv
	    done
	done
    done
    # with word
    test_file=edited_data/${word}_many_many/ptb.train.${word}.fixedwordtest.txt
    python ptb_error_analysis.py --train_file_path=edited_data/${word}_many_many/ptb.train.no.${word}.txt --word_test_file_path=${test_file} --new_word=${word} --model_save_path=${word}_withword_train_logs/${word}/pre_fine/ --output_prefix=withword_wordpresent_
    test_file=edited_data/error_falsepositive_test.txt
    python ptb_error_analysis.py --train_file_path=edited_data/${word}_many_many/ptb.train.no.${word}.txt --word_test_file_path=${test_file} --new_word=${word} --model_save_path=${word}_withword_train_logs/${word}/pre_fine/ --output_prefix=withword_wordpresent_
done
