#!/bin/bash

#make folder to hold data
mkdir COMMERCIALFRAGMENTS_SAMPLED_DRUGSANALOGS.FILTER

#process each drug
for drug in $(ls drugs)
do

	mkdir COMMERCIALFRAGMENTS_SAMPLED_DRUGSANALOGS.FILTER/$drug

	#process each learning rate
	for lr in 0.01 0.005 0.001 0.0005 0.0001
	do

		mkdir COMMERCIALFRAGMENTS_SAMPLED_DRUGSANALOGS.FILTER/$drug/$lr

		#process each epoch
		for epoch in 4 9 14 19
		do

			#filters compounds using rdkit smartqueries
			python removeCpdsBasedOnSmartsQ.py COMMERCIALFRAGMENTS_SAMPLED_DRUGSANALOGS.FILTER/$drug/$lr/$epoch.smi smartsq.txt out_tmp_filtered.smi "False"

			#calculate the similarity of generated compounds with respect to query
			python avalone_similaritycalculation.py $drug/$drugs out_tmp_filtered.smi COMMERCIALFRAGMENTS_SAMPLED_DRUGSANALOGS.FILTER/$drug/$lr/${epoch}.smi

			#remove intermediate files
			rm out_tmp.smi drug.smi out_tmp_filtered.smi

		done
	done
done
