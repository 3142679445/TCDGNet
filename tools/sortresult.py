import os
import shutil

import pandas
def sortresult(filepath):
    data=pandas.read_csv(filepath,header=None)
    if len(data)>3:
        sorted_data = data.sort_values(by=data.columns[3], ascending=False)
        top3 = []
        for i in range(3):
            modelname = "model" + str(int(sorted_data.iloc[i][7])) + ".pth"
            top3.append(modelname)
        directory = filepath[:-7] + "models/"
        for filename in os.listdir(directory):
            if filename not in top3:
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        sorted_data.to_csv(filepath[:-4] + '_sort.csv', header=False, index=False)
# sortresult(filepath)

if __name__=="__main__":
    filepath="/home/ldw/Csegment/runs/16/acc.csv"
    sortresult(filepath)