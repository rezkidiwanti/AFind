import AFSignalProcessing as SP
import numpy as np

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

def test_program(list_data, status):
    #DATA
    data_path_AF = 'Data TEST/'
    data_ecg_AF = list_data
    # Signal Processing anda Feature Calculation
    featureAF = SP.AF_SP(data_path_AF, data_ecg_AF, signal_type = status)
    return featureAF
