import os
import numpy as np
import pandas as pd
from lmfit import Parameters
import matplotlib.pyplot as plt
import mqnmr_functions as mqn
import random


# needed 3rd-party packages: Numpy, Pandas, Matplotlib, Lmfit
# use >> pip install [package name] << to install it


styledUnicodeDict = {
    "success": "\033[1;38;5;82m\u2714\033[0m",   # Light neon green checkmark
    "failure": "\033[1;38;5;196m\u2717\033[0m",  # Light neon red cross
    "process": "\033[1;38;5;87m\u25B6\uFE0E\033[0m"  # Light neon yellow arrow
}

# change working directiory to location of this file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# array of files to be evaluated
file_str_array = ["BP_PEG12-T-3_305"]

# directory of these files relative to the path of this file
data_dir_str = "./PEG12-T-3/"

# name of the folder where results are saved
save_dir_folder_name = "fit_results"
save_dir_subfolder_ending = ""

# a1 comp variation limits between 0 and 1
a1_limits = [0.05, 0.8]
# max time at which the dq signal is cut off - should be roughly at which it is smaller than 0.01
dq_cutoff_after_ms = 120
# step width of the a1 parameter variation
a1_step = 0.01
# number of fit repetitions with random starting parameters
num_rand_reps = 5
# number of points (counted backwards) for the tail fitting
x_last_points_for_tail = 25
# factorial boundary for estimating fit errors based on the minimal residual
fact_boundary = 1.1


def init_fit_params_randomized(a1_set, tv):
    # fit parameter setup using the lmfit package
    # https://lmfit.github.io/lmfit-py/
    # much better and more stable+advanced than SciPy

    # components should be ordered such that a1 has the highest elastic effectiveness, a2 follows afterwards etc
    # a4 component is used for the tail
    # all values for a3 are set as "do not vary them" in order to have a fit with only two elastic components

    fit_params = Parameters()

    fit_params.add('a1', value=a1_set, vary=False)
    fit_params.add('a4', value=tv[0], min=0.9*tv[0], max=1.1*tv[0])
    fit_params.add('a2', expr='1-a1-a4')
    fit_params.add('a3', value=0, vary=False)

    fit_params.add('rdc1', value=random.uniform(0.04, 0.3), min=0.02, max=0.3)
    fit_params.add('rdc2', value=random.uniform(
        0.01, 0.06), min=0.002, max=0.1)
    fit_params.add('rdc3', value=0, vary=False)

    fit_params.add('t21', value=random.uniform(3, 15), min=3, max=15)
    fit_params.add('t22', value=random.uniform(15, 80), min=10, max=80)
    fit_params.add('t23', value=1, vary=False)
    fit_params.add('t24', value=tv[1], min=0.8*tv[1], max=1.2*tv[1])

    fit_params.add('b1', value=random.uniform(1.5, 2.0), min=1.1, max=2.0)
    fit_params.add('b2', value=random.uniform(1.2, 2.0), min=1.0, max=2.0)
    fit_params.add('b3', value=0, vary=False)

    # pseudo parameters used for constructing inequality relationships between parameters
    fit_params.add('delta_t_1', expr='t22-t21', min=0, vary=True)
    fit_params.add('delta_t_2', expr='t23-t22', min=0, vary=True)
    fit_params.add('delta_t_3', expr='t24-t23', min=0, vary=True)
    fit_params.add('delta_rdc_1', expr='rdc1-rdc2', min=0, vary=True)
    fit_params.add('delta_rdc_2', expr='rdc2-rdc3', min=0, vary=True)
    fit_params.add('delta_b_1', expr='b1-b2', min=0, vary=True)
    fit_params.add('delta_b_2', expr='b2-b3', min=0, vary=True)

    return fit_params

#########################################################################
#########################################################################
### everything below here should not be changed without being careful ###
#########################################################################
#########################################################################

# loss function with separate return of residuals


def loss_fun_split(parDict, t, ISMQ, IDQ):
    tdq = t[0:len(IDQ)]
    res_isum = (ISMQ - mqn.ismq_fit_fun(t, parDict))
    res_dq = (len(t)/len(tdq))*(IDQ - mqn.idq_fit_fun(tdq, parDict))/max(IDQ)

    return res_dq, res_isum

# loss function with concatenated return of residuals
# lmfit needs this, unfortunately


def loss_fun(parDict, t, ISMQ, IDQ):

    tdq = t[0:len(IDQ)]
    res_isum = (ISMQ - mqn.ismq_fit_fun(t, parDict))
    res_dq = (IDQ - mqn.idq_fit_fun(tdq, parDict))/max(IDQ)

    res_total = np.append(res_isum, res_dq)

    return res_total

################################################
############ actual run starts here ############
################################################


print(
    f'\t\n {styledUnicodeDict["process"]} Started MQ-NMR mult-fit routine with the following array: {file_str_array}')

# loop ovrt all files given in file_str_array
for fileStr in file_str_array:

    print(
        f'\t\n {styledUnicodeDict["process"]} Started subroutine for: {fileStr}')

    save_dir_str = f"{save_dir_folder_name}/{fileStr}{save_dir_subfolder_ending}/"

    if not os.path.exists(f"{save_dir_folder_name}"):
        os.mkdir(f"{save_dir_folder_name}")

    if not os.path.exists(f"{save_dir_folder_name}/{fileStr}{save_dir_subfolder_ending}"):
        os.mkdir(f"{save_dir_folder_name}/{fileStr}{save_dir_subfolder_ending}")
        print(
            f'\t\n {styledUnicodeDict["process"]} Result subdirectory not found. A new one was created \n')
    else:
        print(
            f'\t\n {styledUnicodeDict["process"]} Result subdirectory found. Existing files will be overwritten \n')

    try:

        # convert minispec data to a pandas dataframe
        df = pd.read_csv(data_dir_str + fileStr + ".txt",
                         sep="\t", names=['t', 'iref', 'idq', 'imag'])

        # cut and normalize data
        data = mqn.prepare_data(df, dq_cutoff=dq_cutoff_after_ms)

        # run the fit routine
        fitresult_dict = mqn.a1_fit_routine_randomized(
            data=data,
            loss_fun=loss_fun,
            loss_fun_split=loss_fun_split,
            init_fit_params=init_fit_params_randomized,
            a1_stepnumber=int((max(a1_limits) - min(a1_limits))/a1_step),
            rand_reps=num_rand_reps,
            tail_index=x_last_points_for_tail,
            a1_limits=a1_limits
        )

        # determine best fit parameters based on the residual surface
        best_opt_params = fitresult_dict['out_matrix'][fitresult_dict['min_idx']].params
        [best_sum_res, best_dq_res] = mqn.calc_fit_res(data, best_opt_params)

        # get fit errors according to the normalized factor given here
        # 1.1 (=10%) is usually fine, but may need adjustments based on the individual samples
        # inspect results and decide
        fit_result_df, cutted_matrix = mqn.calculate_fit_errors(
            fitresult_dict, fact_boundary)

        # save results as csv
        fit_result_df.to_csv(save_dir_str + fileStr + '_fitparams' + '.txt', header=['parameter', 'fit_value', 'lb_fit', 'ub_fit'],
                             sep='\t')

        result_df, exp_df, best_fitresult_plot_c = mqn.create_result_df(
            data=data,
            best_opt_params=best_opt_params,
            file_str=fileStr
        )

        # individual components saved as csv for easy data plots
        exp_df.to_csv(save_dir_str + fileStr + '_expData' + '.txt',
                      header=['time', 'I-sum', 'I-DQ', 'I-sum-fit', 'I-DQ-fit', 'comp1-fit', 'comp2-fit', 'comp3-fit'], sep='\t')

        plt.savefig(save_dir_str + "global_fit_" + fileStr + ".jpg", dpi=400)

        # create a residual surface plot
        a1_plot = mqn.create_a1_resplot(
            a1_list=fitresult_dict['a1_list'],
            res_dic=fitresult_dict['res_dic'],
            file_str=fileStr,
            res_indexmarks=[fitresult_dict['min_idx'],],
            factorial_boundary=1.1
        )
        plt.savefig(save_dir_str + "a1_surface_" + fileStr + ".jpg", dpi=400)

        # fit residual plot
        resplot = mqn.create_resplot(
            data, best_sum_res, best_dq_res, exp_df, fileStr)
        plt.savefig(save_dir_str + "res_plot_" + fileStr + ".jpg", dpi=400)

        # plot confidence intervals
        conf_boundary_fitplot = mqn.plot_confidence_interval(
            data,  cutted_matrix, best_opt_params)
        plt.savefig(save_dir_str + "conf_boundary_fitplot_" +
                    fileStr + ".jpg", dpi=400)

        print(
            f'\t\n {styledUnicodeDict["success"]} FINISHED SAMPLE WITH FILE STRING: ', fileStr)
        print('--------------------------------------')

    except Exception as err:
        print(
            f'\t\n {styledUnicodeDict["failure"]} The following exception occured at sample ${fileStr}: \n')
        print(err)
        print('--------------------------------------')
