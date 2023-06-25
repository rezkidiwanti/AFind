import AFSignalProcessing as SP
from matplotlib import pyplot as plt
import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

#RUNNING ONLY FOR GETTING SPLITTED DATA TRAIN
def plott(title, AF, Normal, pnjng):
    # RR INTERVAL
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title(title + 'Max RR')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][0], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][0], 'g.', markersize=8)

    plt.subplot(2, 2, 2)
    plt.title(title + 'Min RR')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][1], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][1], 'g.', markersize=8)

    plt.subplot(2, 2, 3)
    plt.title(title + 'Mean RR')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][2], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][2], 'g.', markersize=8)

    plt.subplot(2, 2, 4)
    plt.title(title + 'STD RR')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][3], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][3], 'g.', markersize=8)

    # QRS WIDTH
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title(title + 'Max QRS')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][4], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][4], 'g.', markersize=8)

    plt.subplot(2, 2, 2)
    plt.title(title + 'Min QRS')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][5], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][5], 'g.', markersize=8)

    plt.subplot(2, 2, 3)
    plt.title(title + 'Mean QRS')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][6], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][6], 'g.', markersize=8)

    plt.subplot(2, 2, 4)
    plt.title(title + 'STD QRS')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][7], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][7], 'g.', markersize=8)

    # TP Duration
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title(title + 'Max TP')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][8], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][8], 'g.', markersize=8)

    plt.subplot(2, 2, 2)
    plt.title(title + 'Min TP')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][9], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][9], 'g.', markersize=8)

    plt.subplot(2, 2, 3)
    plt.title(title + 'Mean TP')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][10], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][10], 'g.', markersize=8)

    plt.subplot(2, 2, 4)
    plt.title(title + 'STD TP')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][11], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][11], 'g.', markersize=8)

    # PQ DURATION
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title(title + 'Max PQ')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][12], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][12], 'g.', markersize=8)

    plt.subplot(2, 2, 2)
    plt.title(title + 'Min PQ')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][13], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][13], 'g.', markersize=8)

    plt.subplot(2, 2, 3)
    plt.title(title + 'Mean PQ')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][14], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][14], 'g.', markersize=8)

    plt.subplot(2, 2, 4)
    plt.title(title + 'STD PQ')
    for i in range(pnjng):
        plt.plot(i + 1, AF[i][15], 'r.', markersize=8)
        plt.plot(i + 1, Normal[i][15], 'g.', markersize=8)

def Run_program(list_data, status):
    #DATA
    data_path_AF = 'Data AF/'
    data_ecg_AF = list_data
    data_path_N = 'Data NORMAL/'
    data_ecg_N = list_data


    # Signal Processing anda Feature Calculation
    featureAF = SP.AF_SP(data_path_AF, data_ecg_AF, signal_type = status)
    featureN = SP.AF_SP(data_path_N, data_ecg_N, signal_type = status)

    return featureAF, featureN

def Make_data_train():
    list_data = []
    for i in range(200):
        list_data.append(str(i+1))
    AF, Normal = Run_program(list_data, 1)
    AF2, Normal2 = Run_program(list_data, 2)

    AF = AF + AF2
    Normal = Normal + Normal2
    plott(' ', AF, Normal, 400)
    # # Before Transform
    # fields = ['maxRR', 'minRR', 'meanRR', 'stdevRR', 'maxQRS', 'minQRS', 'meanQRS', 'stdevQRS', 'maxTP', 'minTP',
    #           'meanTP', 'stdevTP',
    #           'maxPQ', 'minPQ', 'meanPQ', 'stdevPQ']
    # rows = AF
    # with open('AFIB.csv', 'w', newline="") as f:
    #     write = csv.writer(f)
    #     write.writerow(fields)
    #     write.writerows(rows)
    # fields = ['maxRR', 'minRR', 'meanRR', 'stdevRR', 'maxQRS', 'minQRS', 'meanQRS', 'stdevQRS', 'maxTP', 'minTP',
    #           'meanTP', 'stdevTP',
    #           'maxPQ', 'minPQ', 'meanPQ', 'stdevPQ']
    # rows = Normal
    # with open('NORMAL.csv', 'w', newline="") as f:
    #     write = csv.writer(f)
    #     write.writerow(fields)
    #     write.writerows(rows)

    NAF = AF.copy()
    NN = Normal.copy()
    Min = 0
    for i in range (len(NAF)):
        for j in range(16):
            if NAF[i][j] <= Min :
                Min = NAF[i][j]
            if NN[i][j] <= Min :
                Min = NN[i][j]

    # diff = abs(1-Min)
    for i in range(len(NAF)):
        for j in range(16):
            # NAF[i][j] = NAF[i][j] + diff
            NAF[i][j] = NAF[i][j] * NAF[i][j]

            # NN[i][j] = NN[i][j] + diff
            NN[i][j] = NN[i][j] * NN[i][j]

    plott('New ', NAF, NN, 400)
    # # After Transform
    # fields = ['maxRR','minRR','meanRR','stdevRR','maxQRS','minQRS','meanQRS','stdevQRS','maxTP','minTP','meanTP','stdevTP',
    #                       'maxPQ','minPQ','meanPQ','stdevPQ']
    # rows = NAF
    # with open('NEW AFIB.csv', 'w', newline="") as f:
    #     write = csv.writer(f)
    #     write.writerow(fields)
    #     write.writerows(rows)
    # fields = ['maxRR','minRR','meanRR','stdevRR','maxQRS','minQRS','meanQRS','stdevQRS','maxTP','minTP','meanTP','stdevTP',
    #                       'maxPQ','minPQ','meanPQ','stdevPQ']
    # rows = NN
    # with open('NEW NORMAL.csv', 'w', newline="") as f:
    #     write = csv.writer(f)
    #     write.writerow(fields)
    #     write.writerows(rows)
    return AF,Normal,NAF, NN

def Train(dpAF, dpN):
    df_afib = pd.read_csv(dpAF, skiprows=[1])  # read csv file
    df_normal = pd.read_csv(dpN, skiprows=[1])  # read csv file
    df_afib['label'] = ('AFIB')  # add label
    df_normal['label'] = ('NORMAL')  # add label
    new_df = pd.concat([df_afib, df_normal], axis=0)  # concatenate 2 dataframe
    new_df = new_df.sample(frac=1).reset_index(drop=True)  # shuffle dataframe
    X = new_df.drop(['label'], axis=1)  # drop label
    y = new_df['label']  # label
    models = {
        'RandomForestClassifier': RandomForestClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier()
    }
    kf = KFold(n_splits=10, shuffle=True, random_state=42)  # split data n split bisa di ganti sesuai kebutuhan
    for name, model in models.items():
        pipeline = Pipeline(steps=[('model', model)])
        i = 0
        print(name)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            conf = confusion_matrix(y_test, y_pred)
            TP = conf[0, 0]
            FP = conf[0, 1]
            FN = conf[1, 0]
            TN = conf[1, 1]
            accuracy = (TP + TN) / (TP + FP + FN + TN)
            sensitivity = (TP) / (TP + FN)
            specificity = (TN) / (TN + FP)

            print(f'accuracy: {round(accuracy, 2)}')
            print(f'sensitivity: {round(sensitivity, 2)}')
            print(f'specificity: {round(specificity, 2)}')
            print(' ')
            # pickle.dump(pipeline, open('model_' + name + '_' + str(i) + '.pkl', 'wb'))
            i = i + 1
        print('-' * 100)


AF,Normal,NAF, NN = Make_data_train()
# print('BEFORE TRANSFORM')
# Train('AFIB.csv','NORMAL.csv')
# print('AFTER TRANSFORM')
# Train('NEW AFIB.csv', 'NEW NORMAL.csv')

# plott(' ', AF, Normal, 400)
# plott('New ', NAF, NN, 400)
plt.show()
# AF = [[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]]
# N = [[1,3,5,7,9],[1,3,5,7,9],[1,3,5,7,9],[1,3,5,7,9],[1,3,5,7,9]]
# print(AF)
# print(N)
# NAF = AF.copy()
# NN = N.copy()
# diff = 1.0
# for i in range(len(NAF)):
#     if i > 0:
#         for j in range(5):
#             NAF[i][j] = NAF[i][j] + diff
#             NAF[i][j] = NAF[i][j] * NAF[i][j]
#
#             NN[i][j] = NN[i][j] + diff
#             NN[i][j] = NN[i][j] * NN[i][j]
#
# print(NAF)
# print(NN)