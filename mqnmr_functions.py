import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from lmfit import minimize, report_fit
from scipy.optimize import curve_fit

colorDict = {
    "Rich Black": "#001219",
    "Blue Sapphire": "#005f73",
    "Viridian Green": "#0a9396",
    "Middle Blue Green": "#94d2bd",
    "Medium Champagne": "#e9d8a6",
    "Gamboge": "#ee9b00",
    "Alloy Orange": "#ca6702",
    "Rust": "#bb3e03",
    "Rufous": "#ae2012",
    "Ruby Red": "#9b2226"
}

colorDictPNG = {
    "Dark Blue": "#161E49",
    "Medium Blue": "#295C77",
    "Light Blue": "#45B9BC",
    "Salmon Red": "#F66A49"
}

styledUnicodeDict = {
    "success": "\033[1;38;5;82m\u2714\033[0m",   # Light neon green checkmark
    "failure": "\033[1;38;5;196m\u2717\033[0m",  # Light neon red cross
    "process": "\033[1;38;5;87m\u25B6\uFE0E\033[0m"  # Light neon yellow arrow
}


def tail_fun(t, a, t2):
    return a * np.exp(-(t / t2))


def al_fun(t, rdc):
    return 1 - np.exp(-(2.375 * t * rdc) ** 1.5) * np.cos(
        3.663 * rdc * t)  # + (1-np.exp(-(2.375*t*0.9*rdc)**1.5)*np.cos(3.663*0.9*rdc*t))


def rlx_fun(t, t2, b):
    return np.exp(-(t / t2) ** b)


def ismq_fit_fun(t, par_dic):
    a1, a2, a3, a4 = par_dic['a1'], par_dic['a2'], par_dic['a3'], par_dic['a4']
    t21, t22, t23, t24 = par_dic['t21'], par_dic['t22'], par_dic['t23'], par_dic['t24']
    b1, b2, b3 = par_dic['b1'], par_dic['b2'], par_dic['b3']

    return a1 * rlx_fun(t, t21, b1) + a2 * rlx_fun(t, t22, b2) + a3 * rlx_fun(t, t23, b3) + a4 * rlx_fun(t, t24, b=1)


def idq_fit_fun(t, par_dic):
    a1, a2, a3, a4 = par_dic['a1'], par_dic['a2'], par_dic['a3'], par_dic['a4']
    t21, t22, t23, t24 = par_dic['t21'], par_dic['t22'], par_dic['t23'], par_dic['t24']
    b1, b2, b3 = par_dic['b1'], par_dic['b2'], par_dic['b3']
    rdc1, rdc2, rdc3 = par_dic['rdc1'], par_dic['rdc2'], par_dic['rdc3']

    c1 = rlx_fun(t, t21, b1) * al_fun(t, rdc1)
    c2 = rlx_fun(t, t22, b2) * al_fun(t, rdc2)
    c3 = rlx_fun(t, t23, b3) * al_fun(t, rdc3)

    return 0.5 * (a1 * c1 + a2 * c2 + a3 * c3)


def idq_comp_fun(t, par_dic):
    a1, a2, a3, a4 = par_dic['a1'], par_dic['a2'], par_dic['a3'], par_dic['a4']
    t21, t22, t23, t24 = par_dic['t21'], par_dic['t22'], par_dic['t23'], par_dic['t24']
    b1, b2, b3 = par_dic['b1'], par_dic['b2'], par_dic['b3']
    rdc1, rdc2, rdc3 = par_dic['rdc1'], par_dic['rdc2'], par_dic['rdc3']

    c1 = 0.5 * a1 * rlx_fun(t, t21, b1) * al_fun(t, rdc1)
    c2 = 0.5 * a2 * rlx_fun(t, t22, b2) * al_fun(t, rdc2)
    c3 = 0.5 * a3 * rlx_fun(t, t23, b3) * al_fun(t, rdc3)

    return [c1, c2, c3]


def print_progressbar(iteration, total, prefix='‚', suffix='', decimals=1, length=50, print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = "\033[1;38;5;87m" + "\u2588" * filled_length + \
        "\033[0m" + '-' * (length - filled_length)
    print(f'\r   {prefix} |{bar}| {percent}% {suffix}', end=print_end)

    if iteration == total:
        print()


def prepare_data(df, dq_cutoff):
    df['isum'] = df.iref + df.idq
    df['idq'] /= df.isum[0]
    df['isum'] /= df.isum[0]
    # cut dq at df.t > x milliseconds
    IDQ = df.loc[df.t < dq_cutoff, 'idq'].to_numpy()
    ISMQ = df['isum'].to_numpy()
    t = df['t'].to_numpy()

    return t, ISMQ, IDQ


def a1_fit_routine(data, loss_fun, loss_fun_split, init_fit_params, a1_stepnumber, a1_limits, tail_index=20):
    t, ISMQ, IDQ = data

    a1_list = np.linspace(
        start=a1_limits[0], stop=a1_limits[1], num=a1_stepnumber)
    a1_smq_residuals = []
    a1_dq_residuals = []
    a1_fitresults = []

    tailvals = curve_fit(
        tail_fun, t[-tail_index:], ISMQ[-tail_index:], p0=[0.40, 500])
    tailvals = tailvals[0]
    print(
        f"\t\n {styledUnicodeDict['process']} Estimated tail fraction based on last {tail_index} points: \n\t Fraction: {round(100 * tailvals[0],2)} % \n\t T2: {round(tailvals[1],2)} ms \n")

    for ia, a1 in enumerate(a1_list):
        print_progressbar(ia, a1_stepnumber, prefix='Progress:',
                          suffix='Complete', length=50)

        fpars_init = init_fit_params(a1, tailvals)
        out = minimize(loss_fun, fpars_init, args=(t, ISMQ, IDQ))
        loss_dq, loss_isum = loss_fun_split(
            out.params.valuesdict(), t, ISMQ, IDQ)
        a1_smq_residuals.append(np.linalg.norm(loss_isum)/np.std(loss_isum))
        a1_dq_residuals.append(np.linalg.norm(loss_dq)/np.std(loss_dq))
        a1_fitresults.append(out)

    a1_total_residuals = [(x + y) / 2 for (x, y) in
                          zip(a1_smq_residuals, a1_dq_residuals)]  # list comprehension to calculate sum residuals

    val, idx = min((val, idx) for (idx, val) in enumerate(
        a1_total_residuals))  # get the minimum of total_res

    result_dictionary = {
        "out_matrix": a1_fitresults,
        "min_idx": idx,
        "min_val": val,
        "a1_list": a1_list,
        "res_dic": {
            "total_res": a1_total_residuals,
            "sum_res": a1_smq_residuals,
            "dq_res": a1_dq_residuals,
        }
    }
    return result_dictionary


def a1_fit_routine_randomized(data, loss_fun, loss_fun_split, init_fit_params, a1_stepnumber, rand_reps, a1_limits,
                              tail_index=20):
    t, ISMQ, IDQ = data

    a1_list = np.linspace(
        start=a1_limits[0], stop=a1_limits[1], num=a1_stepnumber)
    a1_smq_residuals = []
    a1_dq_residuals = []
    a1_fitresults = []

    a1_smq_randtemp = []
    a1_dq_randtemp = []
    a1_fitresults_randtemp = []

    tailvals = curve_fit(
        tail_fun, t[-tail_index:], ISMQ[-tail_index:], p0=[0.40, 500])
    tailvals = tailvals[0]
    print(
        f"\t\n {styledUnicodeDict['process']} Estimated tail fraction based on last {tail_index} points: \n\t Fraction: {round(100 * tailvals[0],2)} % \n\t T2: {round(tailvals[1],2)} ms \n")
    counter = 0
    for ia, a1 in enumerate(a1_list):

        ir = 0

        while ir < rand_reps:
            fpars_init = init_fit_params(a1, tailvals)
            out = minimize(loss_fun, fpars_init, args=(t, ISMQ, IDQ))
            loss_dq, loss_isum = loss_fun_split(
                out.params.valuesdict(), t, ISMQ, IDQ)

            a1_smq_randtemp.append(np.linalg.norm(loss_isum))
            a1_dq_randtemp.append(np.linalg.norm(loss_dq))
            a1_fitresults_randtemp.append(out)

            ir += 1
            counter += 1
            print_progressbar(counter, a1_stepnumber * rand_reps, prefix='Progress:', suffix='Complete',
                              length=50)

        a1_total_residuals_randtemp = [
            (x + y) / 2 for (x, y) in zip(a1_smq_randtemp, a1_dq_randtemp)]
        val_rand, idx_rand = min((val, idx) for (
            idx, val) in enumerate(a1_total_residuals_randtemp))

        a1_smq_residuals.append(a1_smq_randtemp[idx_rand])
        a1_dq_residuals.append(a1_dq_randtemp[idx_rand])
        a1_fitresults.append(a1_fitresults_randtemp[idx_rand])

        a1_smq_randtemp = []
        a1_dq_randtemp = []
        a1_fitresults_randtemp = []

    a1_total_residuals = [(x + y) / 2 for (x, y) in
                          zip(a1_smq_residuals, a1_dq_residuals)]  # list comprehension to calculate sum residuals

    val, idx = min((val, idx) for (idx, val) in enumerate(
        a1_total_residuals))  # get the minimum of total_res

    result_dictionary = {
        "out_matrix": a1_fitresults,
        "min_idx": idx,
        "min_val": val,
        "a1_list": a1_list,
        "res_dic": {
            "total_res": a1_total_residuals,
            "sum_res": a1_smq_residuals,
            "dq_res": a1_dq_residuals,
        }
    }
    return result_dictionary


def plot_confidence_interval(data, regression_matrix, best_fit_dict):

    t, sums, dq = data
    t_sampling: list[float] = np.linspace(0, max(t), 501)

    outcome_matrix = np.zeros((len(regression_matrix), len(t_sampling)))
    min_conf_vals: list[float] = []
    max_conf_vals: list[float] = []

    for i, set in enumerate(regression_matrix):
        outcome_matrix[i, :] = idq_fit_fun(
            np.linspace(0, max(t), 501), set.params)

    for column in outcome_matrix.T:
        min_conf_vals.append(min(column))
        max_conf_vals.append(max(column))

    fit_plot = plt.figure(figsize=(7, 5))

    plt.semilogy(t[:len(dq)], dq, 'o', color=colorDictPNG['Dark Blue'],
                 markerfacecolor='none', markersize=4, markeredgewidth=1.2)
    plt.fill_between(t_sampling, y1=min_conf_vals, y2=max_conf_vals,
                     facecolor=colorDictPNG['Light Blue'], alpha=0.55)
    plt.semilogy(t_sampling, idq_fit_fun(np.linspace(0, max(t), 501), best_fit_dict),
                 '-', color=colorDictPNG['Salmon Red'], markerfacecolor='none', linewidth=1.5)

    plt.xlabel('DQ evolution time / ms', fontsize=16)
    plt.ylabel('Intensity / a.u.', fontsize=16)
    plt.tick_params(labelsize=14)
    plt.ylim(0.1, 0.4)
    plt.xlim((-1, 40))
    plt.tight_layout()

    return fit_plot


def create_a1_resplot(a1_list, res_dic, file_str="", res_indexmarks=[], factorial_boundary=1.3):
    a1_resplot = plt.figure(figsize=(7, 5))
    plt.plot(a1_list, res_dic["sum_res"], 's-', markersize=4,
             color=colorDictPNG['Dark Blue'], label=r"$I_{\Sigma MQ}$ res")
    plt.plot(a1_list, res_dic["dq_res"], 'o-', markersize=4,
             color=colorDictPNG['Light Blue'], label=r"$I_{DQ}$ res")
    plt.plot(a1_list, res_dic["total_res"], 'x-', markersize=4,
             color=colorDictPNG['Salmon Red'], label="avg. res")
    plt.hlines(y=factorial_boundary * min(res_dic["total_res"]), xmin=0, xmax=1, linestyle='dotted',
               color=colorDictPNG['Salmon Red'], linewidth=1.5, label='confidence interval')

    for ii in res_indexmarks:
        plt.vlines(x=a1_list[ii], ymin=0, ymax=res_dic["total_res"][ii], linestyle='dashed',
                   color=colorDictPNG['Salmon Red'], linewidth=1, label='_nolabel_')

    plt.title(f'Exemplary residual surface', fontsize=16)
    plt.legend(fontsize=14, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim(0, 2*np.mean(res_dic["total_res"]))
    plt.xlim(0, 1)
    plt.xlabel(r'$a_1$ fraction (single links)', fontsize=16)
    plt.ylabel('RMSE', fontsize=16)
    plt.tight_layout()

    return a1_resplot


def calc_fit_res(data, out):
    t, sums, dq = data
    fit_sum = ismq_fit_fun(t, out.valuesdict())
    fit_dq = idq_fit_fun(t[:len(dq)], out.valuesdict())

    res_sum = 100 * (sums - fit_sum) / sums
    res_dq = 100 * (dq - fit_dq) / dq

    return [res_sum, res_dq]


def create_result_df(data, best_opt_params, file_str=""):
    t, ISMQ, IDQ = data
    keyList = best_opt_params.keys()
    key_list = []
    value_list = []
    err_list = []

    for k in keyList:
        key_list.append(k)
        value_list.append(best_opt_params[k] * 1)
        try:
            err_list.append(best_opt_params[k].stderr * 1)
        except:
            err_list.append('nan')

    result_df = pd.DataFrame(list(zip(key_list, value_list, err_list)), columns=[
                             'parameter', 'value', 'std-err'])

    l_isum = ISMQ.tolist()
    l_idq = IDQ.tolist()
    l_fit_sum = ismq_fit_fun(t, best_opt_params.valuesdict()).tolist()
    l_fit_dq = idq_fit_fun(t, best_opt_params.valuesdict()).tolist()
    [comp1, comp2, comp3] = idq_comp_fun(t, best_opt_params.valuesdict())
    l_comp1 = comp1.tolist()
    l_comp2 = comp2.tolist()
    l_comp3 = comp3.tolist()

    exp_df = pd.DataFrame(list(zip(t, l_isum, l_idq, l_fit_dq, l_fit_sum, l_comp1, l_comp2, l_comp3)),
                          columns=['time', 'I-sum', 'I-DQ', 'I-sum-fit', 'I-DQ-fit', 'comp1-fit', 'comp2-fit',
                                   'comp3-fit'])

    complete_fit_result = plt.figure(figsize=(14, 5))
    plt.semilogy(t, l_isum, 'ko', markerfacecolor='none',
                 markeredgewidth=1.5, markersize=6, label=r'$I_{\Sigma MQ}$')
    plt.semilogy(t[:len(IDQ)], l_idq, 'ks', markerfacecolor='none',
                 markeredgewidth=1.5, markersize=6, label=r'$I_{DQ}$')
    plt.semilogy(t, l_fit_sum, '-',
                 color=colorDictPNG['Salmon Red'], linewidth=2, label='global_fit')
    plt.semilogy(t, l_fit_dq, '-',
                 color=colorDictPNG['Salmon Red'], linewidth=2, label='_nolegend_')
    plt.semilogy(t, l_comp1, '--',
                 color=colorDictPNG['Light Blue'], linewidth=1, label='SL')
    plt.semilogy(t, l_comp2, '-.',
                 color=colorDictPNG['Light Blue'], linewidth=1, label='DL')
    plt.semilogy(t, l_comp3, ':',
                 color=colorDictPNG['Light Blue'], linewidth=1, label='HOC')
    plt.xlabel('DQ evolution time / ms', fontsize=16)
    plt.ylabel('Norm. intensity / a.u.', fontsize=16)
    plt.ylim(0.001, 1)
    plt.xlim(-2, 300)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)
    plt.title(f'Simultaneous fit', fontsize=16)
    plt.tight_layout()
    return result_df, exp_df, complete_fit_result


def create_resplot(data, rs, rdq, exp_df, file_str=""):
    t, ISUM, IDQ = data

    res_plot, (ax2) = plt.subplots(1, 1, figsize=(7, 5))

    ax2.plot(t, rs, 'o-', color=colorDictPNG['Dark Blue'],
             markerfacecolor='none', label=r'$I_{\Sigma MQ}$ res')
    ax2.plot(t[:len(rdq)], rdq, 'o-', color=colorDictPNG['Light Blue'],
             markerfacecolor='none', label=r'$I_{DQ}$ res')
    ax2.set_xlabel('DQ evolution time / ms', fontsize=16)
    ax2.set_ylabel('Norm. res. / %', fontsize=16)
    ax2.set_ylim(-25, 25)
    ax2.set_xlim(0, 150)
    ax2.legend(fontsize=14, loc='upper right')
    ax2.grid('which=major', linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=14)
    plt.title('Percentual deviations of model from data', fontsize=16)
    plt.tight_layout()
    res_plot.align_ylabels()

    return res_plot


def calculate_fit_errors(result_dic, factorial_boundary=1.3):
    out_matrix = result_dic['out_matrix']
    total_res = result_dic['res_dic']['total_res']
    min_idx = result_dic['min_idx']

    minval = total_res[min_idx]
    boundary = factorial_boundary * minval

    try:
        lb_idx = np.where(total_res[:min_idx] > boundary)[0][-1]
    except IndexError:
        lb_idx = 0
    try:
        ub_idx = min_idx + np.where(total_res[min_idx:] > boundary)[0][0]
    except IndexError:
        ub_idx = -1

    cutted_out_matrix = out_matrix[lb_idx: ub_idx + 1]

    fit_error_dic = {
        "parameter": [],
        "fit_value": [],
        "lb_fit": [],
        "ub_fit": []
    }

    cutted_matrix_dic = {key: [] for key in out_matrix[0].params.keys()}

    for m in cutted_out_matrix:
        for key in m.params.keys():
            cutted_matrix_dic[key].append(m.params[key] * 1)

    for key in out_matrix[0].params.keys():
        fit_error_dic['parameter'].append(key)
        fit_error_dic['fit_value'].append(out_matrix[min_idx].params[key] * 1)
        fit_error_dic['lb_fit'].append(min(cutted_matrix_dic[key]))
        fit_error_dic['ub_fit'].append(max(cutted_matrix_dic[key]))

    return pd.DataFrame.from_dict(fit_error_dic), cutted_out_matrix
