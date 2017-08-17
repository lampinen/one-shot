#!/bin/bash
declare -a word_arr=("bonuses" "explained" "strategist" "entry" "rice" "rolled" "marketers" "immune" "cowboys" "borrow")

## now loop through the above array
for new_word in "${word_arr[@]}"
do
  echo Now running $new_word
  mkdir ${new_word}_train_logs
  i=1
  python ptb_word_lm.py --model=large --data_path=raw_data --train_file_path=edited_data/ptb.train.no.${new_word}.txt --word_train_file_path=edited_data/ptb.train.${new_word}.${i}wordtrain.txt --word_test_file_path=edited_data/ptb.train.${new_word}.fixedwordtest.txt --new_word=${new_word} --save_path=${new_word}_train_logs --reload_pre=False --centroid_approach=False --result_log_file=${i}_results_log.csv
    python ptb_word_lm.py --model=large --data_path=raw_data --train_file_path=edited_data/ptb.train.no.${new_word}.txt --word_train_file_path=edited_data/ptb.train.${new_word}.${i}wordtrain.txt --word_test_file_path=edited_data/ptb.train.${new_word}.fixedwordtest.txt --new_word=${new_word} --save_path=${new_word}_train_logs --reload_pre=True --centroid_approach=True --result_log_file=${i}_centroid_results_log.csv
  for i in `seq 2 10`
  do
    python ptb_word_lm.py --model=large --data_path=raw_data --train_file_path=edited_data/ptb.train.no.${new_word}.txt --word_train_file_path=edited_data/ptb.train.${new_word}.${i}wordtrain.txt --word_test_file_path=edited_data/ptb.train.${new_word}.fixedwordtest.txt --new_word=${new_word} --save_path=${new_word}_train_logs --reload_pre=True --centroid_approach=False --result_log_file=${i}_results_log.csv
    python ptb_word_lm.py --model=large --data_path=raw_data --train_file_path=edited_data/ptb.train.no.${new_word}.txt --word_train_file_path=edited_data/ptb.train.${new_word}.${i}wordtrain.txt --word_test_file_path=edited_data/ptb.train.${new_word}.fixedwordtest.txt --new_word=${new_word} --save_path=${new_word}_train_logs --reload_pre=True --centroid_approach=True --result_log_file=${i}_centroid_results_log.csv
  done
done
