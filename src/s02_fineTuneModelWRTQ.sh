#!/bin/bash

mkdir COMMERCIALFRAGMENTS_FINETUNED_MODELS
#========================================================
for drug in $(ls drugs)
do

mkdir COMMERCIALFRAGMENTS_FINETUNED_MODELS/$drug

#learning rates
for lr in 0.01 0.005 0.001 0.0005 0.0001
do

mkdir COMMERCIALFRAGMENTS_FINETUNED_MODELS/$drug/$lr

python build_models_finetune.py CommercialFrags.smi.process COMMERCIALFRAGMENTS_GLOBAL_MODEL/keras_char_rnn.49.h5 drugs/$drug 20 4 $lr COMMERCIALFRAGMENTS_FINETUNED_MODELS/$drug/$lr

done
done
#========================================================
