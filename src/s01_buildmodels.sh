#!/bin/bash

mkdir COMMERCIALFRAGMENTS_GLOBAL_MODEL

# Inputs
# 1) Input database (plain text file containing smiles)
# 2) Directory to hold the models
# 3) number of epochs
# 4) sequence length
 
python build_models.py CommercialFrags.smi.process COMMERCIALFRAGMENTS_GLOBAL_MODEL 50 32
