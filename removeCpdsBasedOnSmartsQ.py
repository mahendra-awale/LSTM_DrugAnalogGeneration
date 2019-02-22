import sys
from rdkit import Chem
from rdkit.Chem import SaltRemover


class SMARTmatcher:

    # Constructor of class#####################################
    def __init__(self, infile, smartfile, outfile, processtrue):

        self._infile = infile
        self._smartfile = smartfile
        self._outfile = outfile
        self._smartPatterns = {}
        self._noofmolsread = 0
        self._noofmolswritten = 0
        self._process = processtrue

    # pass the smartpattern file###############################
    def parse_smart_queries(self, smartfile):

        datareader = open(smartfile, "r")
        smartsMol_Tag = {}

        for line in datareader.readlines():
            line = line.strip()
            splitline = line.split(" ")
            try:
                mol = Chem.MolFromSmarts(splitline[0])
                if mol is None:
                    print(splitline[1])
                    continue
                else:
                    smartsMol_Tag[splitline[1]] = mol
            except:
                continue
        return smartsMol_Tag

    # parse the smiles########################################
    def parseSmiles(self, smiles):

        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.RemoveHs(mol)
            Remover = SaltRemover.SaltRemover()
            mol = Remover.StripMol(mol)
            Chem.SanitizeMol(mol)
            Chem.SetHybridization(mol)
            Chem.SetAromaticity(mol)
            return mol
        except:
            return None

    # matched the smart########################################
    def match_smart_queries(self, smartsQueries, mol):

        try:
            result = {}
            for qtag in smartsQueries:
                qsmartMol = smartsQueries[qtag]
                all_matches = mol.GetSubstructMatches(qsmartMol)
                result[qtag] = len(all_matches)
            return result
        except:
            None

    ##########################################################
    def getDictKeysAsString(self, d, delim):
        out = ""
        for key in d:
            out = out + key + delim
        return out[:-1]

    ##########################################################
    def getDictValsAsString(self, d, delim):
        out = ""
        for key in d:
            out = out + str(d[key]) + delim
        return out[:-1]

    ##########################################################
    def getDictKeyValsAsString(self, d, delim1, delim2):
        out = ""
        for key in d:
            out = out + (str(key) + delim1 + str(d[key])) + delim2
        return out[:-1]

    ##########################################################
    def updateLastLog(self):

        print("=========Below smart queries were used==========")
        for key in self._smartPatterns:
            print(key)
        print("=================================================")

    ##########################################################
    def parseLineAndGetMol(self, line, process):

        linesplit = line.split(" ")
        try:
            mol = Chem.MolFromSmiles(linesplit[0])
            if process == 1:
                mol = self.parseSmiles(linesplit[0])
            return mol
        except:
            return None

    ##########################################################
    def formatOutput(self, line, smartres, outputstyle):

        try:
            if outputstyle == 1:
                output = self.getDictKeyValsAsString(smartres, ":", " ")
                output = line + " " + output
                return output
            else:
                output = self.getDictValsAsString(smartres, ";")
                output = line + " " + output
                return output
        except:
            return "None"

    # Run it##################################################
    def runIt(self):

        try:
            self._smartPatterns = self.parse_smart_queries(self._smartfile)
        except:
            print("Problem in reading smarts query file.")
            print("Exiting for now.")
            exit()

        if len(self._smartPatterns) == 0:
            print("No smart queries provided")
            print("Exiting for now.")
            exit()

        try:
            datareader = open(self._infile, "r")
            datawriter = open(self._outfile, "w")
            outdata = {}
            while True:

                line = datareader.readline()
                if line is "":
                    break

                line = line.strip()
                linesplit = line.split(" ")
                smi = linesplit[0]
                tag = "noname"
                if len(linesplit) > 1:
                    tag = linesplit[1]

                mol = self.parseLineAndGetMol(smi, self._process)

                if mol is None:
                    continue

                if mol.GetNumAtoms() < 4:
                    continue

                res = self.match_smart_queries(self._smartPatterns, mol)
                smi = Chem.MolToSmiles(mol)

                if res is None:
                    continue

                failed = False
                for key in res:
                    if res[key] > 0:
                        failed = True

                if failed:
                    continue

                outdata[line] = smi+" "+tag
            datareader.close()

            totalcpdspass = len(outdata)
            for key in  outdata:
                datawriter.write(outdata[key]+"_"+str(totalcpdspass)+"\n")
            datawriter.close()
            print("END")

        except Exception as er:
            print(er)
            print("Problem in reading/writing database file.")
            print("Exiting for now.")
            exit()


sm = SMARTmatcher(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
sm.runIt()
