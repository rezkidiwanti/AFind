import pandas as pd
import pickle
import AFSignalProcessing
import pickle
import json

loaded_model = pickle.load(open('model_ecg_adaboost_tuned.sav', 'rb'))


def predict(lines, status):
    extraction_all = AFSignalProcessing.make_fiture(lines, status)
    fitur_n_json = []
    for x in [extraction_all]:
        dictnya = {}
        for i, j in zip(x[0], x[1]):
            dictnya[i] = j
        fitur_n_json.append(dictnya)

    df_test = pd.DataFrame(fitur_n_json)

    df_inference = df_test.drop(
        columns=['maxRR', 'maxQRS', 'minQRS', 'meanQRS', 'stdevQRS', 'maxTP'])

    prediksi = loaded_model.predict_proba(df_inference.values)
    if prediksi.argmax() == 1:
        return {'predict': 'AF', 'confidence': prediksi[0][prediksi.argmax()]}
    else:
        return {'predict': 'Normal', 'confidence': prediksi[0][prediksi.argmax()]}