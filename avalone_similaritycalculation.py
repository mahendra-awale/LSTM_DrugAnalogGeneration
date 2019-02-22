from rdkit.Avalon import pyAvalonTools
from rdkit import Chem
import numpy as np
import scipy.spatial.distance
from operator import itemgetter
import sys

# ================================
def read_file(infile):
    smiles = []
    names = []

    with open(infile, "r") as reader:
        for line in reader:
            line = line.strip()
            linesplit = line.split(" ")
            smi = linesplit[0]
            smiles.append(smi)

            if len(linesplit) > 1:
                names.append(linesplit[1])
            else:
                names.append("NoName")
    return smiles, names


# ================================
def calculte_avalone(smi, nbits):
    try:
        mol = Chem.MolFromSmiles(smi)
        fp = pyAvalonTools.GetAvalonFP(mol, nbits)
        return np.asarray(fp, dtype=float)
    except:
        return "None"


# ================================
def query_AvaloneSimToDatabase(qsmi, database_smis, database_names, outputfile):
    qavalone = calculte_avalone(qsmi, 1024)
    if qavalone is "None":
        return None

    output = {}

    for i in range(len(database_smis)):
        dbavalone = calculte_avalone(database_smis[i], 1024)

        if dbavalone is "None":
            continue

        try:
            jaccard = scipy.spatial.distance.jaccard(qavalone, dbavalone)
            output[database_smis[i] + " " + database_names[i]] = jaccard
        except:
            continue

    output = sorted(output.items(), key=itemgetter(1))
    writer = open(outputfile, "w")
    for i in range(len(output)):
        writer.write(output[i][0] + " " + str(output[i][1]) + "\n")
    writer.close()

qfile = sys.argv[1]
dbfile = sys.argv[2]
outfile = sys.argv[3]

qsmis, qnames = read_file(qfile)
dbsmis, dbnames = read_file(dbfile)

query_AvaloneSimToDatabase(qsmis[0], dbsmis, dbnames, outfile)