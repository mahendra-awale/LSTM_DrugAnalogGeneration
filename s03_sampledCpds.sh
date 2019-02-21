#!/bin/bash

mkdir COMMERCIALFRAGMENTS_SAMPLED_DRUGSANALOGS

for drug in $(ls drugs)
do

mkdir COMMERCIALFRAGMENTS_SAMPLED_DRUGSANALOGS/$drug
drug_smi=$(head -n 1 drugs/$drug)

for lr in 0.01 0.005 0.001 0.0005 0.0001
do

mkdir COMMERCIALFRAGMENTS_SAMPLED_DRUGSANALOGS/$drug/$lr

for epoch in $(seq 4 5 19)
do

#sampled the compounds
python /2015/mahendra/Documents/NovartisProject2018/Analysis-1Feb18/holefilers/lstm_models/sample_compounds.py CommercialFrags.smi.process COMMERCIALFRAGMENTS_FINETUNED_MODELS/$drug/$lr/keras_char_rnn.$epoch.h5 COMMERCIALFRAGMENTS_SAMPLED_DRUGSANALOGS/$drug/$lr/${epoch}.smi "numpychoice" "0.5" 200000 $drug_smi

done
done
done
