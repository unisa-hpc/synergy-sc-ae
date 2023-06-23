# This is Python script to predict frequency of miniweather application by using the selected ML algorithm

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
from sklearn.linear_model import (LinearRegression,
                                  Lasso)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import os
import sys

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
default_clk = int(sys.argv[1])

clk_file = f"{script_dir}/../gpu-freq.csv" 
training_file_normalized_loc = f"{script_dir}/../../models-validation/data/training-data/"  # frequency is not normalized
test_file_normalized_loc = f"{script_dir}/features-normalized/"  # frequency is not normalized
predictions_dir = f"{script_dir}/predictions"

os.makedirs(predictions_dir, exist_ok=True)

training_kernel_name = ['merged_ArithLocalMixed.csv', 'merged_ArithMixedUnitOp.csv', 'merged_ArithMixedUnitType.csv',
                        'merged_ArithSingleUnit.csv', 'merged_GlobalMemory.csv', 'merged_GlobalMemory2.csv',
                        'merged_L2Unit.csv', 'merged_LocalMemory.csv', 'merged_Stencil.csv']
test_kernel_name = ['miniWeather_mpi_parallelfor_features.csv']

Nconfig_per_case = 1
fig_cnt = 1
Energy_saving_value = [0.25, 0.50, 0.75]

def func_ML_Predict_CLK(x_feature, df_x_obj, model, y_features, df_clk, num):
    clk_list = []
    model.fit(x_feature, df_x_obj.values.ravel())
    for kernel_i in range(0, num):
        df_test = y_features.iloc[[kernel_i]]
        df_feature_per_kernel = pd.concat([df_test]*len(df_clk), ignore_index=True)
        y_model_test_feature = pd.concat([df_clk, df_feature_per_kernel], axis=1)
        predict_temp = model.predict(y_model_test_feature)
        predict_df = pd.DataFrame(data=predict_temp)  # prediction: dataframe

        idx_min = predict_df.idxmin().values[0]
        predict_clk = y_model_test_feature.iloc[idx_min]['core-freq']
        clk_list.append(int(predict_clk))
    return clk_list


def func_ES_clk_prediction(x_feature, df_x_obj, model, y_features, df_clk, num,
                           metric_values  # objective
                           ):
    clk_prediction_list = []
    loc_freq_min_energy_list = []
    loc_freq_default_list = []
    model.fit(x_feature, df_x_obj.values.ravel())

    for kernel_i in range(0, num):
        df_test = y_features.iloc[[kernel_i]]
        df_feature_per_kernel = pd.concat([df_test]*len(df_clk), ignore_index=True)
        y_model_test_feature = pd.concat([df_clk, df_feature_per_kernel], axis=1)
        predict_temp = model.predict(y_model_test_feature)
        predict_df = pd.DataFrame(data=predict_temp, index=y_model_test_feature.index.copy())  # prediction: dataframe

        # =================================
        idx_default_clk = y_model_test_feature['core-freq'].loc[
            lambda x: x == default_clk].index  # index of default clk
        default_energy_pre = predict_df.loc[idx_default_clk].values[0]  # energy prediction with default clk
        min_energy_pre = predict_df.min().values[0]  # min_energy in the prediction list
        idx_min_energy = predict_df.idxmin().values[0]
        clk_min_energy = y_model_test_feature['core-freq'].loc[
            idx_min_energy]  # clk for minimizing energy prediction
        # =================================
        freq_array = y_model_test_feature['core-freq'].to_numpy()
        loc_freq_min_energy = np.where(freq_array == clk_min_energy)[0][0]
        loc_freq_min_energy_list.append(loc_freq_min_energy)
        loc_freq_default = np.where(freq_array == default_clk)[0][0]
        loc_freq_default_list.append(loc_freq_default)
        # =================================
        for value_i in metric_values:
            energy_expect = default_energy_pre * (1 - value_i) + min_energy_pre * value_i

            idx_clk_prediction = (predict_df.iloc[
                                  loc_freq_default:loc_freq_min_energy + 1] - energy_expect).abs().sort_values(
                predict_df.columns[0])[:1].index
            clk_prediction = y_model_test_feature['core-freq'].loc[idx_clk_prediction].values[0]
            clk_prediction_list.append(int(clk_prediction))

    return clk_prediction_list, loc_freq_min_energy_list, loc_freq_default_list


def func_PL_clk_prediction(x_feature, df_x_obj, model, y_features, df_clk, num,
                           metric_values, loc_freq_min_energy_list, loc_freq_default_list):
    clk_prediction_list = []
    model.fit(x_feature, df_x_obj.values.ravel())

    for kernel_i in range(0, num):
        df_test = y_features.iloc[[kernel_i]]
        df_feature_per_kernel = pd.concat([df_test]*len(df_clk), ignore_index=True)
        y_model_test_feature = pd.concat([df_clk, df_feature_per_kernel], axis=1)
        predict_temp = model.predict(y_model_test_feature)
        predict_df = pd.DataFrame(data=predict_temp, index=y_model_test_feature.index.copy())  # prediction: dataframe

        # =================================
        idx_default_clk = y_model_test_feature['core-freq'].loc[
            lambda x: x == default_clk].index  # index of default clk
        default_time_pre = predict_df.loc[idx_default_clk].values[0]  # energy prediction with default clk

        min_time_pre = predict_df.iloc[loc_freq_min_energy_list[kernel_i]].values[0]
        # =================================

        # =================================
        for value_i in metric_values:
            time_expect = default_time_pre * (1 - value_i) + min_time_pre * value_i

            idx_clk_prediction = (predict_df.iloc[
                                  loc_freq_default_list[kernel_i]:loc_freq_min_energy_list[kernel_i] + 1] - time_expect).abs().sort_values(
                predict_df.columns[0])[:1].index
            clk_prediction = y_model_test_feature['core-freq'].loc[idx_clk_prediction].values[0]
            clk_prediction_list.append(int(clk_prediction))


    return clk_prediction_list

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ===========================================================================================================
    # training input data: features, frequency, objectives

    df_list_feature = []
    df_list_core_clk = []
    df_list_time = []
    df_list_energy = []
    df_list_edp = []
    df_list_ed2p = []
    df_list_miniweather = []

    Time_clk_list = []
    Energy_clk_list = []
    EDP_clk_list = []
    ED2P_clk_list = []
    ES_clk_list = []
    PL_clk_list = []

    for kernel_i in range(0, len(training_kernel_name)):  # len(training_kernel_name)

        df_train_data = pd.read_csv(training_file_normalized_loc + training_kernel_name[kernel_i])

        df_list_feature.append(df_train_data.loc[:, 'mem_gl':'flt_add'])
        df_list_core_clk.append(df_train_data[['core-freq']])
        df_list_time.append(df_train_data[['kernel-time [s]']])
        df_list_energy.append(df_train_data[['mean-energy [J]']])
        df_list_edp.append(df_train_data[['mean-edp']])
        df_list_ed2p.append(df_train_data[['mean-ed2p']])

    df_train_features = pd.concat(df_list_feature)
    df_train_core_clk = pd.concat(df_list_core_clk)
    df_train_obj_time = pd.concat(df_list_time)
    df_train_obj_energy = pd.concat(df_list_energy)
    df_train_obj_edp = pd.concat(df_list_edp)
    df_train_obj_ed2p = pd.concat(df_list_ed2p)

    del df_list_feature[:], df_list_core_clk[:], df_list_time[:], df_list_energy[:], df_list_edp[:], df_list_ed2p[:]

    # ===========================================================================================================
    # prediction input data: features, model, frequency

    for kernel_i in range(0, len(test_kernel_name)):
        df_test_data = pd.read_csv(test_file_normalized_loc + test_kernel_name[kernel_i])

        if not df_test_data.empty:
            df_list_miniweather.append(df_test_data.loc[:, 'kernel_name':'flt_add'])

    df_miniweather = pd.concat(df_list_miniweather)
    kernel_num = len(df_miniweather)   # the total number of kernels

    df_miniweather_features = df_miniweather.loc[:, 'mem_gl':'flt_add']

    # ===========================================================================================================
    # import clk
    df_gpu_clk = pd.read_csv(clk_file, usecols=['core-freq'])

    # ===========================================================================================================
    # Model and predict optimal frequency for each metric
    x_model_train_feature = pd.concat([df_train_core_clk, df_train_features], axis=1)
    # ================= ML prediction and print the predicted frequency for each test benchmarks

    Time_clk_list = func_ML_Predict_CLK(x_model_train_feature, df_train_obj_time, Lasso(alpha=1.0),
                                        df_miniweather_features, df_gpu_clk, kernel_num)

    Energy_clk_list = func_ML_Predict_CLK(x_model_train_feature, df_train_obj_energy, SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
                                          df_miniweather_features, df_gpu_clk, kernel_num)

    EDP_clk_list = func_ML_Predict_CLK(x_model_train_feature, df_train_obj_edp, RandomForestRegressor(),
                                       df_miniweather_features, df_gpu_clk, kernel_num)

    ED2P_clk_list = func_ML_Predict_CLK(x_model_train_feature, df_train_obj_ed2p, LinearRegression(),
                                        df_miniweather_features, df_gpu_clk, kernel_num)
    # ===========================================================================================================
    ES_column_name = ['clk_es_25', 'clk_es_50', 'clk_es_75']
    ES_clk_list, Loc_Freq_Min_Energy, Loc_Freq_Default = \
        func_ES_clk_prediction(x_model_train_feature, df_train_obj_energy,
                               RandomForestRegressor(),
                               df_miniweather_features, df_gpu_clk, kernel_num, Energy_saving_value)  # objective

    ES_clk_list_sublist = [ES_clk_list[i:i + 3] for i in range(0, len(ES_clk_list), 3)]
    ES_clk_df = pd.DataFrame(ES_clk_list_sublist, columns=ES_column_name)
    # ES_clk_df.to_csv('output_miniweather_frequency_prediction_ES.csv')

    # ===========================================================================================================
    PL_column_name = ['clk_pl_25', 'clk_pl_50', 'clk_pl_75']
    PL_clk_list = func_PL_clk_prediction(x_model_train_feature, df_train_obj_time, Lasso(alpha=1.0),
                                         df_miniweather_features, df_gpu_clk, kernel_num, Energy_saving_value,
                                         Loc_Freq_Min_Energy, Loc_Freq_Default)  # objective

    PL_clk_list_sublist = [PL_clk_list[i:i + 3] for i in range(0, len(PL_clk_list), 3)]
    PL_clk_df = pd.DataFrame(PL_clk_list_sublist, columns=PL_column_name)
    # PL_clk_df.to_csv('output_miniweather_frequency_prediction_PL.csv')
    # ===========================================================================================================
    # generate output file, .csv
    df_output = df_miniweather[['kernel_name']]
    df_output.insert(1, "core_clk_time", Time_clk_list, True)
    df_output.insert(2, "core_clk_energy", Energy_clk_list, True)
    df_output.insert(3, "core_clk_edp", EDP_clk_list, True)
    df_output.insert(4, "core_clk_ed2p", ED2P_clk_list, True)

    df_merge = pd.concat([ES_clk_df,PL_clk_df], axis=1)
    df_merge_all = pd.concat([df_output,df_merge], axis=1)  #, ignore_index=True)

    row_start = 0
    for kernel_i in range(0, len(test_kernel_name)):
        name = test_kernel_name[kernel_i].replace("_features", "")
        df_data = pd.read_csv(test_file_normalized_loc + test_kernel_name[kernel_i])
        if not df_data.empty:
            row_num = len(df_data)
            df_data_to_save = df_merge_all.iloc[row_start:row_start+row_num]
            row_start = row_start+row_num
            df_data_to_save.to_csv(f"{predictions_dir}/{name}", index=False)
