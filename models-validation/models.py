# This is Python script to train models using different ML algorithms.
# Objectives: Time, Energy, EDP, ED2P

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os
from collections.abc import Iterable
from matplotlib.pyplot import figure

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import (LinearRegression,
                                  Lasso)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
data_dir = sys.argv[1]
default_clk = int(sys.argv[2])

Energy_saving_value = [0.25, 0.50, 0.75]

training_file_normalized_loc = f"{script_dir}/{data_dir}/training-data/"  # frequency is not normalized
test_file_normalized_loc = f"{script_dir}/{data_dir}/testing-data/"

savepath_predictions = f"{script_dir}/predictions/"
savepath_validation = f"{script_dir}/predictions/benchmarks-errors/"

algorithms_file = f"{savepath_predictions}/algorithms.txt"


training_kernel_name = ['merged_ArithLocalMixed.csv', 'merged_ArithMixedUnitOp.csv', 'merged_ArithMixedUnitType.csv',
                        'merged_ArithSingleUnit.csv', 'merged_GlobalMemory.csv', 'merged_GlobalMemory2.csv',
                        'merged_L2Unit.csv', 'merged_LocalMemory.csv', 'merged_Stencil.csv']
test_case = ['BitCompression', 'BlackScholes', 'BoxBlur', 'Ftle', 'Geometricmean', 'Kmeans_fp32', 'Kmeans_fp64',
             'Knn', 'LinearRegression_fp32', 'LinearRegression_fp64', 'Matrix_mul', 'Matrix_transpose', 'MedianFilter',
             'MerseTwister', 'MolecularDynamics', 'Nbody_local_mem', 'ScalarProduct_NDRange_fp32', 'Sinewave',
             'Sobel3', 'VectorAddition_fp32', 'VectorAddition_fp64', 'VectorAddition_int32',
             'VectorAddition_int64']
x_tick = ['BC', 'BS', 'BB', 'Ftle', 'G-mean', 'Kmeans_fp32', 'Kmeans_fp64', 'Knn', 'LR_fp32', 'LR_fp64', 'Mat_mul',
          'Mat_T', 'MF', ' MT', 'MD', 'Nbody', 'SP_fp32', 'Sinewave', 'Sobel',
          'Vec_Add_fp32', 'Vec_Add_fp64', 'Vec_Add_int32', 'Vec_Add_int64']

# generate from energy modeling with RandomForest regression
loc_freq_min_energy = [117,113,116,117,113,114,114,101,101,111,104,104,113,117,117,117,122,113,117,122,117,117,122]
loc_freq_default = [157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157]


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

# Defining MAPE function
def APE(Y_actual, Y_Predicted):
    ape = np.abs((Y_actual - Y_Predicted) / Y_actual) * 100
    return ape

def func_PL_clk_prediction(x_feature, df_x_obj,  # training data
                           df_y_core_clk, df_y_features,  # testing features
                           y_measurement, y_measurement_df,  # testing measurement
                           predictor_dict,  # ML predictors
                           metric_values,  # objective
                           n_config_per_case
                           ):
    clk_prediction_list = []
    metric_prediction_list = []
    metric_measurement_list = []
    for case_i in range(0, len(test_case)):  # model each test benchmarks ------- len(test_case)

        case_start = n_config_per_case * case_i
        case_end = n_config_per_case + n_config_per_case * case_i
        df_test_core_clk_per_case = df_y_core_clk[case_start:case_end]
        df_test_feature_per_case = df_y_features[case_start:case_end]
        y_model_test_feature = pd.concat([df_test_core_clk_per_case, df_test_feature_per_case], axis=1)

        for name, predictor in predictor_dict:
            model = make_pipeline(predictor)
            model.fit(x_feature, df_x_obj.values.ravel())
            predict_temp = model.predict(y_model_test_feature)
            predict_df = pd.DataFrame(data=predict_temp,
                                      index=y_model_test_feature.index.copy())  # prediction: dataframe
             # =================================
            idx_default_clk = y_model_test_feature['core-freq'].loc[
                lambda x: x == default_clk].index  # index of default clk
            default_time_pre = predict_df.loc[idx_default_clk].values[0]  # energy prediction with default clk

            min_time_pre = predict_df.iloc[loc_freq_min_energy[case_i]].values[0]

            # =================================
            for value_i in metric_values:
                time_expect = default_time_pre * (1 - value_i) + min_time_pre * value_i

                idx_clk_prediction = (predict_df.iloc[
                                      loc_freq_min_energy[case_i]:loc_freq_default[case_i] + 1] - time_expect).abs().sort_values(
                    predict_df.columns[0])[:1].index
                clk_prediction = y_model_test_feature['core-freq'].loc[idx_clk_prediction]

                # print(
                #     f'{test_case[case_i]:10} ==> PL_{value_i} ==> {name:15} ==> frequency prediction is {int(clk_prediction)}')
                clk_prediction_list.append(int(clk_prediction))

    return clk_prediction_list


def func_ES_clk_prediction(x_feature, df_x_obj,  # training data
                           df_y_core_clk, df_y_features,  # testing features
                           y_measurement, y_measurement_df,  # testing measurement
                           predictor_dict,  # ML predictors
                           metric_values,  # objective
                           n_config_per_case
                           ):
    clk_prediction_list = []
    metric_prediction_list = []
    metric_measurement_list = []
    for case_i in range(0, len(test_case)):  # model each test benchmarks ------- len(test_case)

        case_start = n_config_per_case * case_i
        case_end = n_config_per_case + n_config_per_case * case_i
        df_test_core_clk_per_case = df_y_core_clk[case_start:case_end]
        df_test_feature_per_case = df_y_features[case_start:case_end]
        y_model_test_feature = pd.concat([df_test_core_clk_per_case, df_test_feature_per_case], axis=1)

        for name, predictor in predictor_dict:
            model = make_pipeline(predictor)
            model.fit(x_feature, df_x_obj.values.ravel())
            predict_temp = model.predict(y_model_test_feature)
            predict_df = pd.DataFrame(data=predict_temp,
                                      index=y_model_test_feature.index.copy())  # prediction: dataframe
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
            loc_freq_default = np.where(freq_array == default_clk)[0][0]
            # =================================
            for value_i in metric_values:
                energy_expect = default_energy_pre * (1 - value_i) + min_energy_pre * value_i

                idx_clk_prediction = (predict_df.iloc[
                                      loc_freq_min_energy:loc_freq_default + 1] - energy_expect).abs().sort_values(
                    predict_df.columns[0])[:1].index
                clk_prediction = y_model_test_feature['core-freq'].loc[idx_clk_prediction]

                # print(
                #     f'{test_case[case_i]:10} ==> ES_{value_i} ==> {name:15} ==> frequency prediction is {int(clk_prediction)}')
                clk_prediction_list.append(int(clk_prediction))

    return clk_prediction_list


def func_PlotErrorinBar(x_tick, mse_dict, objective, max=70):
    index = np.arange(len(x_tick))  # the label locations
    barWidth = 0.2  # set width of bar
    barHeight_adjust = 0.15
    multiplier = 0

    color_list = ['w', 'lightgrey', 'grey', 'w']  # for each ML algorithm
    hatch_list = ['..', '///', 'xxx', '++']  # for each ML algorithm

    fig, axs = plt.subplots(1, 1, figsize=(12, 4), gridspec_kw={'wspace': 0.05})  # sharex=False, sharey=True,

    for Model, measurement in mse_dict.items():
        offset = barWidth * multiplier
        rects = axs.bar(index + offset, measurement, barWidth, color=color_list[multiplier], edgecolor='k',
                        hatch=hatch_list[multiplier], label=Model)
        # axs.bar_label(rects, padding=3)
        multiplier += 1

    axs.set_xticks(index + barWidth, x_tick)
    axs.set_xticklabels(x_tick, fontsize=12, rotation=70, ha='right', rotation_mode='anchor')
    axs.set_ylabel('Absolute Percentage Error [%]', fontsize=12)
    # axs.set_yticklabels()

    axs.set_ylim(bottom=0)
    axs.legend(loc='upper center', fontsize=12, ncol=multiplier)

    axs.grid(axis='y')
    # axs[0].yaxis.set_major_formatter(PercentFormatter(1))
    # axs[0].set_title('')
    plt.savefig(savepath_validation + objective + '_APE.pdf', dpi=plt.gcf().dpi, bbox_inches='tight')
    plt.close(fig)


def func_MachineLearning_Regression(x_feature, df_x_obj,  # training data
                                    df_y_core_clk, df_y_features,  # testing features
                                    y_measurement, y_measurement_df,  # testing measurement
                                    predictor_dict,  # ML predictors
                                    objective,  # objective
                                    n_config_per_case
                                    ):
    clk_prediction_list = []
    metric_prediction_list = []
    metric_measurement_list = []
    for case_i in range(0, len(test_case)):  # model each test benchmarks ------- len(test_case)
        case_start = n_config_per_case * case_i
        case_end = n_config_per_case + n_config_per_case * case_i
        df_test_core_clk_per_case = df_y_core_clk[case_start:case_end]
        df_test_feature_per_case = df_y_features[case_start:case_end]
        y_model_test_feature = pd.concat([df_test_core_clk_per_case, df_test_feature_per_case], axis=1)

        # # time-model

        for name, predictor in predictor_dict:
            model = make_pipeline(predictor)
            model.fit(x_feature, df_x_obj.values.ravel())
            predict_temp = model.predict(y_model_test_feature)
            predict_df = pd.DataFrame(data=predict_temp)  # prediction: dataframe
            predict_array = np.zeros(np.size(predict_temp))
            for i in range(0, np.size(predict_temp)):
                predict_array[i] = predict_df.iat[i, 0]  # prediction: array

            idx_min = predict_df.idxmin().values[0]

            clk_prediction = y_model_test_feature.iloc[idx_min]['core-freq']
            clk_prediction_list.append(int(clk_prediction))

            label_clk = y_model_test_feature.index[idx_min]
            predict_value = y_measurement_df[case_start:case_end].loc[label_clk].values[0]
            metric_prediction_list.append(predict_value)
            metric_measurement_list.append(y_measurement_df[case_start:case_end].min().values[0])

            ape = APE(y_measurement_df[case_start:case_end].min().values[0], predict_value)
            ape_list.append(ape)

    return ape_list, clk_prediction_list, metric_prediction_list, metric_measurement_list


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    if not os.path.exists(savepath_validation):
        os.makedirs(savepath_validation)

    open(algorithms_file, "w").close()

    # ===========================================================================================================
    # training input data: features, frequency, objectives
    print("Reading datasets...")
    # df = pd.concat([pd.read_csv(file_name.format(i)) for i in range(1, 11)])
    df_list_feature = []
    df_list_core_clk = []
    df_list_time = []
    df_list_energy = []
    df_list_edp = []
    df_list_ed2p = []

    mse_list = []
    ape_list = []

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

    for kernel_i in range(0, len(test_case)):
        df_test_data = pd.read_csv(test_file_normalized_loc + test_case[kernel_i] + '.csv')
        df_test_data = df_test_data.sort_values(['core-freq'], ascending=[True])

        df_list_feature.append(df_test_data.loc[:, 'mem_gl':'flt_add'])
        df_list_core_clk.append(df_test_data[['core-freq']])
        df_list_time.append(df_test_data[['kernel-time [s]']])
        df_list_energy.append(df_test_data[['mean-energy [J]']])
        df_list_edp.append(df_test_data[['mean-edp']])
        df_list_ed2p.append(df_test_data[['mean-ed2p']])


    df_test_features = pd.concat(df_list_feature)
    df_test_core_clk = pd.concat(df_list_core_clk)
    df_test_obj_time = pd.concat(df_list_time)
    df_test_obj_energy = pd.concat(df_list_energy)
    df_test_obj_edp = pd.concat(df_list_edp)
    df_test_obj_ed2p = pd.concat(df_list_ed2p)

    n_config_per_case = len(df_test_core_clk['core-freq'].unique())

    time_true_array = np.zeros(n_config_per_case * len(test_case))
    energy_true_array = np.zeros(n_config_per_case * len(test_case))
    edp_true_array = np.zeros(n_config_per_case * len(test_case))
    ed2p_true_array = np.zeros(n_config_per_case * len(test_case))

    for array_id in range(0, n_config_per_case * len(test_case)):
        time_true_array[array_id] = df_test_obj_time.iat[array_id, 0]
        energy_true_array[array_id] = df_test_obj_energy.iat[array_id, 0]
        edp_true_array[array_id] = df_test_obj_edp.iat[array_id, 0]
        ed2p_true_array[array_id] = df_test_obj_ed2p.iat[array_id, 0]
    # ===========================================================================================================
    # frequency related to each new metric, e.g., ES_25, PL_25, take them as true value
    df_clk_ES_PL = pd.read_csv(test_file_normalized_loc + 'ES_PL_metrics_freq_measurement.csv')

    # ===========================================================================================================
    # ===========================================================================================================
    # machine learning models: Linear, LASSO, SVN, Random forest,

    # # input training features
    # # plot the prediction result
    x_model_train_feature = pd.concat([df_train_core_clk, df_train_features], axis=1)

    # ===========================================================================================================
    # Time
    # ================= time - ML prediction and plot result for each test benchmarks
    print("Generating time models...")

    predictors = [
        ("Linear", LinearRegression()),
        ("Lasso", Lasso(alpha=1.0)),  # alpha=0.005, max_iter=5000
        ("RandomForest", RandomForestRegressor()),
    ]
    ape_list, clk_prediction_list, metric_pre_list, metric_measure_list = \
        func_MachineLearning_Regression(x_model_train_feature, df_train_obj_time,  # training data
                                        df_test_core_clk, df_test_features,  # testing features
                                        time_true_array, df_test_obj_time,  # testing measurement
                                        predictors,  # ML predictors
                                        'Time',  # objective
                                        n_config_per_case)

    # #============= ## time - prediction error analysis

    inter = len(predictors)
    predictor_cnt = 0
    # Error: absolute percentage error
    APE_dict_time = {
        'Linear': ape_list[0:len(ape_list):inter],
        'Lasso': ape_list[1:len(ape_list):inter],
        'RandomForest': ape_list[2:len(ape_list):inter],
    }
    ape_list.clear()
    func_PlotErrorinBar(x_tick, APE_dict_time, 'Time', 10)

    with open(algorithms_file, "a") as f:
        print('MAX_PERF', file=f)

    # Error: MSE, MAPE, RMSE,
    for name, predictor in predictors:
        mse = mean_squared_error(metric_measure_list[predictor_cnt::inter], metric_pre_list[predictor_cnt::inter])
        rmse = math.sqrt(mse)
        mape = mean_absolute_percentage_error(metric_measure_list[predictor_cnt::inter],
                                              metric_pre_list[predictor_cnt::inter])
    
        with open(algorithms_file, "a") as f:
            print(f'{name:15} - RMSE: {rmse:.4f}, MAPE: {mape:.4f}', file=f)
        predictor_cnt = predictor_cnt + 1

    ## time - prediction accuracy analysis

    # ===========================================================================================================
    # Energy

    # ================= Energy - ML prediction and plot result for each test benchmarks
    print("Generating energy models...")

    predictors = [
        ("RandomForest", RandomForestRegressor()),
        ("SVR_RBF", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)),
    ]

    ape_list, clk_prediction_list, metric_pre_list, metric_measure_list = \
        func_MachineLearning_Regression(x_model_train_feature, df_train_obj_energy,  # training data
                                        df_test_core_clk, df_test_features,  # testing features
                                        energy_true_array, df_test_obj_energy,  # testing measurement
                                        predictors,  # ML predictors
                                        'Energy',  # objective
                                        n_config_per_case)

    # #============= ## Energy - prediction error analysis

    inter = len(predictors)
    predictor_cnt = 0
    APE_dict_energy = {
        'RandomForest': ape_list[0:len(ape_list):inter],
        'SVR_RBF': ape_list[1:len(ape_list):inter],
    }
    ape_list.clear()
    func_PlotErrorinBar(x_tick, APE_dict_energy, 'Energy', 30)

    # Error: MSE, MAPE, RMSE,
    with open(algorithms_file, "a") as f:
        print('\nMIN_ENERGY', file=f)

    for name, predictor in predictors:
        mse = mean_squared_error(metric_measure_list[predictor_cnt::inter], metric_pre_list[predictor_cnt::inter])
        rmse = math.sqrt(mse)
        mape = mean_absolute_percentage_error(metric_measure_list[predictor_cnt::inter],
                                              metric_pre_list[predictor_cnt::inter])
        with open(algorithms_file, "a") as f:
            print(f"{name:15} - RMSE: {rmse:.4f}, MAPE: {mape:.4f}", file=f)
        predictor_cnt = predictor_cnt + 1
    ## Energy - prediction accuracy analysis

    # ===========================================================================================================
    # # EDP

    # ================= EDP - ML prediction and plot result for each test benchmarks
    print("Generating EDP models...")

    predictors = [
        ("RandomForest", RandomForestRegressor()),
        ("SVR_RBF", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)),
    ]

    ape_list, clk_prediction_list, metric_pre_list, metric_measure_list = \
        func_MachineLearning_Regression(x_model_train_feature, df_train_obj_edp,  # training data
                                        df_test_core_clk, df_test_features,  # testing features
                                        edp_true_array, df_test_obj_edp,  # testing measurement
                                        predictors,  # ML predictors
                                        'EDP',  # objective
                                        n_config_per_case)

    # #============= ## EDP - prediction error analysis

    inter = len(predictors)
    predictor_cnt = 0
    APE_dict_edp = {
        'RandomForest': ape_list[0:len(ape_list):inter],
        'SVR_RBF': ape_list[1:len(ape_list):inter],
    }
    ape_list.clear()
    func_PlotErrorinBar(x_tick, APE_dict_edp, 'EDP')

    # Error: MSE, MAPE, RMSE,
    with open(algorithms_file, "a") as f:
        print('\nMIN_EDP', file=f)

    for name, predictor in predictors:
        mse = mean_squared_error(metric_measure_list[predictor_cnt::inter], metric_pre_list[predictor_cnt::inter])
        rmse = math.sqrt(mse)
        mape = mean_absolute_percentage_error(metric_measure_list[predictor_cnt::inter],
                                              metric_pre_list[predictor_cnt::inter])
        with open(algorithms_file, "a") as f:
            print(f"{name:15} - RMSE: {rmse:.4f}, MAPE: {mape:.4f}", file=f)
        predictor_cnt = predictor_cnt + 1
    ## EDP - prediction accuracy analysis

    # ===========================================================================================================
    # # ED2P
    # ================= ED2P - ML prediction and plot result for each test benchmarks
    print("Generating ED2P models...")

    predictors = [
        ("Linear", LinearRegression()),
        # ("Lasso", Lasso(alpha=1.0, tol=0.0001)),
        ("RandomForest", RandomForestRegressor()),
        ("SVR_RBF", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)),
    ]

    ape_list, clk_prediction_list, metric_pre_list, metric_measure_list = \
        func_MachineLearning_Regression(x_model_train_feature, df_train_obj_ed2p,  # training data
                                        df_test_core_clk, df_test_features,  # testing features
                                        ed2p_true_array, df_test_obj_ed2p,  # testing measurement
                                        predictors,  # ML predictors
                                        'ED2P',  # objective
                                        n_config_per_case)

    # #============= ## ED2P - prediction error analysis

    inter = len(predictors)
    predictor_cnt = 0
    APE_dict_ed2p = {
        # 'Lasso': ape_list[0:len(ape_list):inter],
        'Linear': ape_list[0:len(ape_list):inter],
        'RandomForest': ape_list[1:len(ape_list):inter],
        'SVR_RBF': ape_list[2:len(ape_list):inter],
    }
    ape_list.clear()
    func_PlotErrorinBar(x_tick, APE_dict_ed2p, 'ED2P')

    # Error: MSE, MAPE, RMSE,
    with open(algorithms_file, "a") as f:
        print('\nMIN_ED2P', file=f)

    for name, predictor in predictors:
        mse = mean_squared_error(metric_measure_list[predictor_cnt::inter], metric_pre_list[predictor_cnt::inter])
        rmse = math.sqrt(mse)
        mape = mean_absolute_percentage_error(metric_measure_list[predictor_cnt::inter],
                                              metric_pre_list[predictor_cnt::inter])
        with open(algorithms_file, "a") as f:
            print(f"{name:15} - RMSE: {rmse:.4f}, MAPE: {mape:.4f}", file=f)
        predictor_cnt = predictor_cnt + 1
    # ED2P - prediction accuracy analysis

    # ===========================================================================================================
    # ES metric
    # ================= ES metric - ML prediction and plot result for each test benchmarks

    print("Generating energy saving predictions...")

    predictors = [
        ("RandomForest", RandomForestRegressor()),
        ("SVR_RBF", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)),
    ]

    # for type, value in itertools.product(['ES'], Energy_saving_value):
    ES_column_name = ['RandomForest_ES_25', 'RandomForest_ES_50', 'RandomForest_ES_75',
                      'SVR_ES_25', 'SVR_ES_50', 'SVR_ES_75']
    if os.path.isfile(f"{savepath_predictions}/frequency_prediction_ES.csv"):
        clk_prediction_df = pd.read_csv(f"{savepath_predictions}/frequency_prediction_ES.csv")

    else:
        clk_prediction_list = func_ES_clk_prediction(x_model_train_feature, df_train_obj_energy,  # training data
                                                     df_test_core_clk, df_test_features,  # testing features
                                                     energy_true_array, df_test_obj_energy,  # testing measurement
                                                     predictors,  # ML predictors
                                                     Energy_saving_value,  # objective
                                                     n_config_per_case)
        clk_prediction_sublist = [clk_prediction_list[i:i + 6] for i in range(0, len(clk_prediction_list), 6)]
        clk_prediction_df = pd.DataFrame(clk_prediction_sublist, columns=ES_column_name)
        clk_prediction_df['kernel'] = test_case
        clk_prediction_df.to_csv(f"{savepath_predictions}/frequency_prediction_ES.csv")

    # ================= ES metric - error analysis
    measure_energy_es_clk_list = []
    predict_energy_es_clk_list = []
    for kernel in test_case:
        df_test_data = pd.read_csv(test_file_normalized_loc + kernel + '.csv')
        #  measurement
        es_clk_array = df_clk_ES_PL.loc[df_clk_ES_PL['kernel'] == kernel, 'es25':'es75'].to_numpy()

        for es_clk in es_clk_array[0]:
            measure_energy_es_clk = df_test_data.loc[df_test_data['core-freq'] == es_clk, 'mean-energy [J]'].values[0]
            measure_energy_es_clk_list.append(measure_energy_es_clk)

    measure_energy_es_clk_sublist = [measure_energy_es_clk_list[i:i + 3] for i in
                                     range(0, len(measure_energy_es_clk_list), 3)]
    measure_energy_es_clk_df = pd.DataFrame(measure_energy_es_clk_sublist, index=test_case,
                                            columns=['es25-energy', 'es50-energy', 'es75-energy'])

    # ES prediction
    for kernel in test_case:
        df_test_data = pd.read_csv(test_file_normalized_loc + kernel + '.csv')
        es_clk_array = clk_prediction_df.loc[clk_prediction_df['kernel'] == kernel,
                       'RandomForest_ES_25':'SVR_ES_75'].to_numpy()

        for es_clk in es_clk_array[0]:
            pre_energy_es_clk = df_test_data.loc[df_test_data['core-freq'] == es_clk, 'mean-energy [J]'].values[0]
            predict_energy_es_clk_list.append(pre_energy_es_clk)

    predict_energy_es_clk_sublist = [predict_energy_es_clk_list[i:i + 6] for i in
                                     range(0, len(predict_energy_es_clk_list), 6)]
    predict_energy_es_clk_df = pd.DataFrame(predict_energy_es_clk_sublist, index=test_case,
                                            columns=ES_column_name)

    #  RMSE, MAPE
    ES_25_RandomForest_rmse = math.sqrt(mean_squared_error(measure_energy_es_clk_df[['es25-energy']].values,
                                                           predict_energy_es_clk_df[['RandomForest_ES_25']].values))
    ES_50_RandomForest_rmse = math.sqrt(mean_squared_error(measure_energy_es_clk_df[['es50-energy']].values,
                                                           predict_energy_es_clk_df[['RandomForest_ES_50']].values))
    ES_75_RandomForest_rmse = math.sqrt(mean_squared_error(measure_energy_es_clk_df[['es75-energy']].values,
                                                           predict_energy_es_clk_df[['RandomForest_ES_75']].values))

    ES_25_RandomForest_mape = mean_absolute_percentage_error(measure_energy_es_clk_df[['es25-energy']].values,
                                                             predict_energy_es_clk_df[['RandomForest_ES_25']].values)
    ES_50_RandomForest_mape = mean_absolute_percentage_error(measure_energy_es_clk_df[['es50-energy']].values,
                                                             predict_energy_es_clk_df[['RandomForest_ES_50']].values)
    ES_75_RandomForest_mape = mean_absolute_percentage_error(measure_energy_es_clk_df[['es75-energy']].values,
                                                             predict_energy_es_clk_df[['RandomForest_ES_75']].values)

    ES_25_SVR_rmse = math.sqrt(mean_squared_error(measure_energy_es_clk_df[['es25-energy']].values,
                                                  predict_energy_es_clk_df[['SVR_ES_25']].values))
    ES_50_SVR_rmse = math.sqrt(mean_squared_error(measure_energy_es_clk_df[['es50-energy']].values,
                                                  predict_energy_es_clk_df[['SVR_ES_50']].values))
    ES_75_SVR_rmse = math.sqrt(mean_squared_error(measure_energy_es_clk_df[['es75-energy']].values,
                                                  predict_energy_es_clk_df[['SVR_ES_75']].values))

    ES_25_SVR_mape = mean_absolute_percentage_error(measure_energy_es_clk_df[['es25-energy']].values,
                                                    predict_energy_es_clk_df[['SVR_ES_25']].values)
    ES_50_SVR_mape = mean_absolute_percentage_error(measure_energy_es_clk_df[['es50-energy']].values,
                                                    predict_energy_es_clk_df[['SVR_ES_50']].values)
    ES_75_SVR_mape = mean_absolute_percentage_error(measure_energy_es_clk_df[['es75-energy']].values,
                                                    predict_energy_es_clk_df[['SVR_ES_75']].values)
    
    with open(algorithms_file, "a") as f:
        print('\nES_25', file=f)
        print(f"RandomForest - RMSE: {ES_25_RandomForest_rmse:.4f}, MAPE: {ES_25_RandomForest_mape:.4f}", file=f)
        print(f"SVR - RMSE: {ES_25_SVR_rmse:.4f}, MAPE {ES_25_SVR_mape:.4f}", file=f)

        print('\nES_50', file=f)
        print(f"RandomForest - RMSE: {ES_50_RandomForest_rmse:.4f}, MAPE: {ES_50_RandomForest_mape:.4f}", file=f)
        print(f"SVR - RMSE: {ES_50_SVR_rmse:.4f}, MAPE {ES_50_SVR_mape:.4f}", file=f)

        print('\nES_75', file=f)
        print(f"RandomForest - RMSE: {ES_75_RandomForest_rmse:.4f}, MAPE: {ES_75_RandomForest_mape:.4f}", file=f)
        print(f"SVR - RMSE: {ES_75_SVR_rmse:.4f}, MAPE {ES_75_SVR_mape:.4f}", file=f)

    # ===========================================================================================================
    # # PL metric
    # ================= PL metric - ML prediction and plot result for each test benchmarks
    print("Generating performance loss predictions...")

    predictors = [
        ("Linear", LinearRegression()),
        ("Lasso", Lasso(alpha=1.0)),  # alpha=0.005, max_iter=5000
        ("RandomForest", RandomForestRegressor()),
    ]

    # for type, value in itertools.product(['ES'], Energy_saving_value):
    PL_column_name = ['Linear_PL_25', 'Linear_PL_50', 'Linear_PL_75',
                      'Lasso_PL_25', 'Lasso_PL_50', 'Lasso_PL_75',
                      'RandomForest_PL_25', 'RandomForest_PL_50', 'RandomForest_PL_75']

    if os.path.isfile(f"{savepath_predictions}/frequency_prediction_PL.csv"):
        clk_prediction_df = pd.read_csv(f"{savepath_predictions}/frequency_prediction_PL.csv")

    else:
        clk_prediction_list = func_PL_clk_prediction(x_model_train_feature, df_train_obj_time,  # training data
                                                     df_test_core_clk, df_test_features,  # testing features
                                                     time_true_array, df_test_obj_time,  # testing measurement
                                                     predictors,  # ML predictors
                                                     Energy_saving_value,  # objective
                                                     n_config_per_case)
        clk_prediction_sublist = [clk_prediction_list[i:i + 9] for i in range(0, len(clk_prediction_list), 9)]
        clk_prediction_df = pd.DataFrame(clk_prediction_sublist, columns=PL_column_name)
        clk_prediction_df['kernel'] = test_case
        clk_prediction_df.to_csv(f"{savepath_predictions}/frequency_prediction_PL.csv")

    # ================= PL metric - error analysis
    measure_time_pl_clk_list = []
    predict_time_pl_clk_list = []
    # PL measurement
    for kernel in test_case:
        df_test_data = pd.read_csv(test_file_normalized_loc + kernel + '.csv')

        pl_clk_array = df_clk_ES_PL.loc[df_clk_ES_PL['kernel'] == kernel, 'pl25':'pl75'].to_numpy()

        for pl_clk in pl_clk_array[0]:
            measure_time_pl_clk = df_test_data.loc[df_test_data['core-freq'] == pl_clk, 'kernel-time [s]'].values[0]
            measure_time_pl_clk_list.append(measure_time_pl_clk)

    measure_time_pl_clk_sublist = [measure_time_pl_clk_list[i:i + 3] for i in
                                   range(0, len(measure_time_pl_clk_list), 3)]
    measure_time_pl_clk_df = pd.DataFrame(measure_time_pl_clk_sublist, index=test_case,
                                          columns=['pl25-time', 'pl50-time', 'pl75-time'])

    # PL prediction
    for kernel in test_case:
        df_test_data = pd.read_csv(test_file_normalized_loc + kernel + '.csv')
        pl_clk_array = clk_prediction_df.loc[clk_prediction_df['kernel'] == kernel,
                       'Linear_PL_25':'RandomForest_PL_75'].to_numpy()

        for pl_clk in pl_clk_array[0]:
            pre_time_pl_clk = df_test_data.loc[df_test_data['core-freq'] == pl_clk, 'kernel-time [s]'].values[0]
            predict_time_pl_clk_list.append(pre_time_pl_clk)

    predict_time_pl_clk_sublist = [predict_time_pl_clk_list[i:i + 9] for i in
                                   range(0, len(predict_time_pl_clk_list), 9)]
    predict_time_pl_clk_df = pd.DataFrame(predict_time_pl_clk_sublist, index=test_case,
                                          columns=PL_column_name)

    #  RMSE, MAPE
    PL_25_Linear_rmse = math.sqrt(mean_squared_error(measure_time_pl_clk_df[['pl25-time']].values,
                                                     predict_time_pl_clk_df[['Linear_PL_25']].values))
    PL_50_Linear_rmse = math.sqrt(mean_squared_error(measure_time_pl_clk_df[['pl50-time']].values,
                                                     predict_time_pl_clk_df[['Linear_PL_50']].values))
    PL_75_Linear_rmse = math.sqrt(mean_squared_error(measure_time_pl_clk_df[['pl75-time']].values,
                                                     predict_time_pl_clk_df[['Linear_PL_75']].values))

    PL_25_Linear_mape = mean_absolute_percentage_error(measure_time_pl_clk_df[['pl25-time']].values,
                                                       predict_time_pl_clk_df[['Linear_PL_25']].values)
    PL_50_Linear_mape = mean_absolute_percentage_error(measure_time_pl_clk_df[['pl50-time']].values,
                                                       predict_time_pl_clk_df[['Linear_PL_50']].values)
    PL_75_Linear_mape = mean_absolute_percentage_error(measure_time_pl_clk_df[['pl75-time']].values,
                                                       predict_time_pl_clk_df[['Linear_PL_75']].values)

    PL_25_Lasso_rmse = math.sqrt(mean_squared_error(measure_time_pl_clk_df[['pl25-time']].values,
                                                    predict_time_pl_clk_df[['Lasso_PL_25']].values))
    PL_50_Lasso_rmse = math.sqrt(mean_squared_error(measure_time_pl_clk_df[['pl50-time']].values,
                                                    predict_time_pl_clk_df[['Lasso_PL_50']].values))
    PL_75_Lasso_rmse = math.sqrt(mean_squared_error(measure_time_pl_clk_df[['pl75-time']].values,
                                                    predict_time_pl_clk_df[['Lasso_PL_75']].values))

    PL_25_Lasso_mape = mean_absolute_percentage_error(measure_time_pl_clk_df[['pl25-time']].values,
                                                      predict_time_pl_clk_df[['Lasso_PL_25']].values)
    PL_50_Lasso_mape = mean_absolute_percentage_error(measure_time_pl_clk_df[['pl50-time']].values,
                                                      predict_time_pl_clk_df[['Lasso_PL_50']].values)
    PL_75_Lasso_mape = mean_absolute_percentage_error(measure_time_pl_clk_df[['pl75-time']].values,
                                                      predict_time_pl_clk_df[['Lasso_PL_75']].values)

    PL_25_RandomForest_rmse = math.sqrt(mean_squared_error(measure_time_pl_clk_df[['pl25-time']].values,
                                                           predict_time_pl_clk_df[['RandomForest_PL_25']].values))
    PL_50_RandomForest_rmse = math.sqrt(mean_squared_error(measure_time_pl_clk_df[['pl50-time']].values,
                                                           predict_time_pl_clk_df[['RandomForest_PL_50']].values))
    PL_75_RandomForest_rmse = math.sqrt(mean_squared_error(measure_time_pl_clk_df[['pl75-time']].values,
                                                           predict_time_pl_clk_df[['RandomForest_PL_75']].values))

    PL_25_RandomForest_mape = mean_absolute_percentage_error(measure_time_pl_clk_df[['pl25-time']].values,
                                                             predict_time_pl_clk_df[['RandomForest_PL_25']].values)
    PL_50_RandomForest_mape = mean_absolute_percentage_error(measure_time_pl_clk_df[['pl50-time']].values,
                                                             predict_time_pl_clk_df[['RandomForest_PL_50']].values)
    PL_75_RandomForest_mape = mean_absolute_percentage_error(measure_time_pl_clk_df[['pl75-time']].values,
                                                             predict_time_pl_clk_df[['Lasso_PL_75']].values)

    with open(algorithms_file, "a") as f:
        print('\nPL_25', file=f)
        print(f"Linear - RMSE: {PL_25_Linear_rmse:.4f}, MAPE: {PL_25_Linear_mape:.4f}", file=f)
        print(f"Lasso - RMSE: {PL_25_Lasso_rmse:.4f}, MAPE: {PL_25_Lasso_mape:.4f}", file=f)
        print(f"RandomForest - RMSE: {PL_25_RandomForest_rmse:.4f}, MAPE: {PL_25_RandomForest_mape:.4f}", file=f)

        print('\nPL_50', file=f)
        print(f"Linear - RMSE: {PL_50_Linear_rmse:.4f}, MAPE: {PL_50_Linear_mape:.4f}", file=f)
        print(f"Lasso - RMSE: {PL_50_Lasso_rmse:.4f}, MAPE: {PL_50_Lasso_mape:.4f}", file=f)
        print(f"RandomForest - RMSE: {PL_50_RandomForest_rmse:.4f}, MAPE: {PL_50_RandomForest_mape:.4f}", file=f)

        print('\nPL_75', file=f)
        print(f"Linear - RMSE: {PL_75_Linear_rmse:.4f}, MAPE: {PL_75_Linear_mape:.4f}", file=f)
        print(f"Lasso - RMSE: {PL_75_Lasso_rmse:.4f}, MAPE: {PL_75_Lasso_mape:.4f}", file=f)
        print(f"RandomForest - RMSE: {PL_75_RandomForest_rmse:.4f}, MAPE: {PL_75_RandomForest_mape:.4f}", file=f)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
