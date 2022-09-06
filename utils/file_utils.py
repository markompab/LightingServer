import csv
import numpy as np
import pandas as pd

class CFileUtils:
    @staticmethod
    def loadCSVArrayNp(path):
        results = np.genfromtxt(path, delimiter=",")
        return results

    @staticmethod
    def loadCSVArrayPandas(path):
        df = pd.read_csv(path)
        return df.to_numpy(df)

    @staticmethod
    def loadCSVArrayCsv(path):
        results = []

        with open(path, 'r') as csvfile:
            # reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
            reader = csv.reader(csvfile)  # retain
            for row in reader:  # each row is a list
                results.append(row)

        return results

    def saveToFile(filepath, data):
        csvfile = open(filepath, 'w')
        csvwriter = csv.writer(csvfile)

        for row in data:
            csvwriter.writerow(row)
