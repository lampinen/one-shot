#!/bin/bash
declare -a word_arr=("bonuses")
# "explained" "strategist" "entry" "rice")
declare -a approach_arr=("opt" "centroid" "opt_centroid" "opt_zero")

## now loop through the above array
for new_word in "${word_arr[@]}"
do
  data_loc=edited_data/${new_word}_many_many/
  for j in `seq 1 10` 
  do 
    
    save_path=result_data/${new_word}_perm${j}_train_logs
    mkdir -p $save_path/{${new_word}/embedding/,tmp}
    for i in `seq 1 10`
    do
      for approach in "${approach_arr[@]}"
      do
	echo Now running $new_word $approach perm $j num_train $i
	python ptb_word_lm.py --model=large --data_path=raw_data --train_file_path=${data_loc}ptb.train.no.${new_word}.txt --word_train_file_path=${data_loc}ptb.train.${new_word}perm${j}.${i}wordtrain.txt --word_test_file_path=${data_loc}ptb.train.${new_word}.fixedwordtest.txt --new_word=${new_word} --save_path=${save_path} --reload_pre=True --model_save_path=${new_word}_train_logs/${new_word}/pre_fine/ --approach=${approach} --result_log_file=${i}_${approach}_results_log.csv
      rm ${save_path}/tmp/*
      done
    done
  done

  save_path=result_data/${new_word}_withword_train_logs/
  mkdir -p $save_path/{${new_word}/embedding/,tmp}
  mkdir -p ${new_word}_withword_train_logs/${new_word}/pre_fine/
  echo Now running $new_word $approach withword
  python ptb_word_lm.py --model=large --data_path=raw_data --train_file_path=${data_loc}ptb.train.nofixedtest.${new_word}.txt --word_train_file_path=${data_loc}ptb.train.${new_word}perm1.10wordtrain.txt --word_test_file_path=${data_loc}ptb.train.${new_word}.fixedwordtest.txt --new_word=${new_word} --save_path=${save_path} --reload_pre=False --model_save_path=${new_word}_withword_train_logs/${new_word}/pre_fine/ --approach=${approach} --result_log_file=_results_log.csv
done
