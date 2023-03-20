import os
import pickle
import matplotlib.pyplot as plt
import pandas

from adaptive_cosim.MSD1Adaptive import MSD1Adaptive
from adaptive_cosim.MSD2Adaptive import MSD2Adaptive
from cosim_msd_utils import CoupledMSD, get_analytical_error
import numpy as np

from fsutils import resource_file_path
from sys_params import X0


def plot_results(results):
    # Instantiate these just for the plotting.
    msd1 = MSD1Adaptive("msd1", True)
    msd2 = MSD2Adaptive("msd2", True)
    sol = CoupledMSD("sol")

    results_s1_s2 = results["s1_s2"]
    results_s2_s1 = results["s2_s1"]
    results_adaptive = results["adap_power"]

    params = results["params"]
    params_string = ', '.join(map(lambda f: "{:.2f}".format(f), params))

    fig, (p1, p2, p3) = plt.subplots(3, sharex=True)

    p1.set_title(f'State Error ({params_string})')

    p1.plot(results_s1_s2.timestamps,
             get_analytical_error(results_s1_s2, msd1, sol, 'v1'),
             label="s1_s2_error_v1")

    p1.plot(results_s2_s1.timestamps,
             get_analytical_error(results_s2_s1, msd1, sol, 'v1'),
             label="s2_s1_error_v1")

    p1.plot(results_adaptive.timestamps,
             get_analytical_error(results_adaptive, msd1, sol, 'v1'),
             label="adaptive_error_v1")

    p1.legend()

    p2.set_title('Fk Extrapolated Error')

    p2.plot(results_s1_s2.timestamps,
             np.abs(results_s1_s2.out_signals[msd1.instanceName][msd1.error_fk_direct]),
             label="s1_s2_fk_error_direct (Actual)")

    p2.plot(results_s1_s2.timestamps,
             np.abs(results_s1_s2.out_signals[msd2.instanceName][msd2.error_fk_indirect]),
             label="s1_s2_fk_error_indirect (Hypo)")

    p2.plot(results_s2_s1.timestamps,
             np.abs(results_s2_s1.out_signals[msd1.instanceName][msd1.error_fk_direct]),
             label="s2_s1_fk_error_direct (Hypo)")

    p2.plot(results_s2_s1.timestamps,
             np.abs(results_s2_s1.out_signals[msd2.instanceName][msd2.error_fk_indirect]),
             label="s2_s1_fk_error_indirect (Actual)")

    p2.legend()

    p3.set_title('Mode')

    p3.plot(results_adaptive.timestamps,
             ["s1->s2" if b else "s2->s1" for b in results_adaptive.out_signals["_ma"][0]],
             label="adaptive_mode")

    p3.legend()


def get_errors(results):
    # Instantiate these just to use their variable names.
    msd1 = MSD1Adaptive("msd1", True)
    sol = CoupledMSD("sol")

    results_s1_s2 = results["s1_s2"]
    results_s2_s1 = results["s2_s1"]
    results_adaptive = results["adap_power"]

    s1_s2_ev1 = get_analytical_error(results_s1_s2, msd1, sol, 'v1')
    s2_s1_ev1 = get_analytical_error(results_s2_s1, msd1, sol, 'v1')
    adaptive_ev1 = get_analytical_error(results_adaptive, msd1, sol, 'v1')

    assert (len(s1_s2_ev1) == len(s2_s1_ev1) == len(adaptive_ev1))

    return s1_s2_ev1, s2_s1_ev1, adaptive_ev1

def score(best_ev1, worst_ev1, adaptive_ev1):
    """
    This scoring function will compute a score for each result.
    The score is divided into three parts:
    - A worst case score -- this means that the adaptive solution error exceeds the worse order error.
    - A guess order score -- this means the adaptive solution error is better than the worse order error, but is not better than the best order error.
    - An best order score -- means the adaptive solution error is better than the best order error.

    The computation of each score is done for each timepoint as follows:
    1. Get best and worst case order.
    2. Check which kind of score needs to be computed.
    2.1 Let ADAPT be the adaptive relative error point
    2.2 Let WO be the worst order relative error point
    2.3 Let BO be the best order relative error point
    3. If worst case: Compute -1.0 * (ADAPT - WO)
    4. If guess case: Compute (WO - ADAPT)
    5. If best case: Compute (BO - ADAPT)
    6. The result is a series of scores

    Aggregation of scores: We take mean and standard deviation of each score, as well as frequency count.

    Notes:
    - The computation of the above scores is only done up to the point where the solution is transient. We use the TOLERANCE parameter to determine this.
    - The computation of each score is done point by point, where each point can be in only one of the three scores.
    - We keep a counter of how many points each kind of score has.
    """

    # Parameterization of the scoring function
    TOLERANCE = 5e-4
    epsilon = 1e-8  # to avoid divisions by zero

    # Find portion of time series that is transient behaviour.
    # We start from the end and move back until one of the errors goes outside tolerance
    for i in range(len(best_ev1) - 1, -1, -1):
        if best_ev1[i] > TOLERANCE or worst_ev1[i] > TOLERANCE or adaptive_ev1[i] > TOLERANCE:
            break

    transient_up_to = i

    # Declare lists for each of the scores: we will just append each value to the score.
    # We don't need to keep the exact information about which exact timestamp has which value
    worst_case_score = list()
    guess_case_score = list()
    best_case_score = list()

    # Start computation of each score per timepoint
    for i in range(transient_up_to):
        # Get best and worst case order.
        best_order_point = min(best_ev1[i], worst_ev1[i])
        worst_order_point = max(best_ev1[i], worst_ev1[i])
        adaptive_point = adaptive_ev1[i]

        # Check which kind of score needs to be computed.
        if best_order_point <= adaptive_point <= worst_order_point:
            # Compute guess case
            guess_case_score.append((worst_order_point - adaptive_point))
        elif adaptive_point > worst_order_point:
            # Compute worst order
            worst_case_score.append(-1.0 * (adaptive_point - worst_order_point))
        else:
            assert (adaptive_point < best_order_point)
            # Compute best case
            best_case_score.append((best_order_point - adaptive_point))

    # Aggregate scores
    worst_case_score_mean = np.nan
    worst_case_score_std = np.nan
    worst_case_score_freq = 0.0

    guess_case_score_mean = np.nan
    guess_case_score_std = np.nan
    guess_case_score_freq = 0.0

    best_case_score_mean = np.nan
    best_case_score_std = np.nan
    best_case_score_freq = 0.0

    if worst_case_score:
        worst_case_score_array = np.array(worst_case_score)
        worst_case_score_array = worst_case_score_array
        worst_case_score_mean = np.mean(worst_case_score_array)
        worst_case_score_std = np.std(worst_case_score_array)
        worst_case_score_freq = len(worst_case_score_array) / transient_up_to

    if guess_case_score:
        guess_case_score_array = np.array(guess_case_score)
        guess_case_score_mean = np.mean(guess_case_score_array)
        guess_case_score_std = np.std(guess_case_score_array)
        guess_case_score_freq = len(guess_case_score_array) / transient_up_to

    if best_case_score:
        best_case_score_array = np.array(best_case_score)
        best_case_score_array = best_case_score_array
        best_case_score_mean = np.mean(best_case_score_array)
        best_case_score_std = np.std(best_case_score_array)
        best_case_score_freq = len(best_case_score_array) / transient_up_to

    best_or_guess_case_freq = best_case_score_freq + guess_case_score_freq

    # DEPRECATED: Old scoring function. Keeping for comparison
    max_error = np.maximum(np.maximum(best_ev1, worst_ev1), adaptive_ev1)
    min_error = np.minimum(np.minimum(best_ev1, worst_ev1), adaptive_ev1)
    total_diff_error = max_error - min_error + epsilon  # Constant for safety
    old_score = np.mean(np.divide((max_error - adaptive_ev1), total_diff_error))

    return [worst_case_score_mean, worst_case_score_std, worst_case_score_freq,
            guess_case_score_mean, guess_case_score_std, guess_case_score_freq,
            best_case_score_mean, best_case_score_std, best_case_score_freq,
            best_or_guess_case_freq,
            old_score, transient_up_to]

def score_universal(best_ev1, worst_ev1, adaptive_ev1):
    worst_case_score_mean = 0.0
    worst_case_score_std = 0.0
    worst_case_score_freq = 0.0
    guess_case_score_mean = 0.0
    guess_case_score_std = 0.0
    guess_case_score_freq = 0.0
    best_case_score_mean = 0.0
    best_case_score_std = 0.0
    best_case_score_freq = 0.0
    best_or_guess_case_freq = 0.0
    old_score = 0.0
    transient_up_to = 0.0

    return [worst_case_score_mean, worst_case_score_std, worst_case_score_freq,
            guess_case_score_mean, guess_case_score_std, guess_case_score_freq,
            best_case_score_mean, best_case_score_std, best_case_score_freq,
            best_or_guess_case_freq,
            old_score, transient_up_to]

def get_errors_and_score(results):
    s1_s2_ev1, s2_s1_ev1, adaptive_ev1 = get_errors(results)
    scoring_results = score(s1_s2_ev1, s2_s1_ev1, adaptive_ev1)
    return [results["params"]] + scoring_results


def apply_fun_parameter_sweep_results(results_dir, fun, max_applications):
    dir_exists = os.path.exists(results_dir)
    assert dir_exists
    res = []
    for filename in os.listdir(results_dir):
        p = os.path.join(results_dir, filename)
        if filename.endswith(".pickle") and filename.startswith("r_"):
            with open(p, "rb") as f:
                results = pickle.load(f)
                res.append(fun(results))
                if max_applications is not None:
                    max_applications = max_applications - 1
                    if max_applications <= 0:
                        return res
    return res


def get_file_name(x0, H, tf, params):
    x0_string = '_'.join(map(lambda f: "{:.2f}".format(f), x0))
    params_string = '_'.join(map(lambda f: "{:.2f}".format(f), params))
    filename = "r_x0_{}_H_{:.2f}_{:.2f}_{}.pickle".format(x0_string, H, tf, params_string)
    return filename


def apply_fun_results(results_dir, x0, H, tf, params, fun):
    filename = get_file_name(x0, H, tf, params)

    filepath = resource_file_path(os.path.join(results_dir, filename))
    assert os.path.exists(filepath)
    with open(filepath, "rb") as f:
        results = pickle.load(f)
        return fun(results)

def save_vars(var,filename):
    filehandler = open(filename, "wb")
    pickle.dump(var, filehandler)
    filehandler.close()

def load_vars(filename):
    FILE = open(filename, 'rb')
    var = pickle.load(FILE)
    FILE.close()
    return var

def export_df_score(scores, given_vars=False):
    df = pandas.DataFrame(scores, columns=['Params',
                                               'Score.WorstCase.mean', 'Score.WorstCase.std', 'Score.WorstCase.freq',
                                               'Score.GuessCase.mean', 'Score.GuessCase.std', 'Score.GuessCase.freq',
                                               'Score.BestCase.mean', 'Score.BestCase.std', 'Score.BestCase.freq',
                                               'Score.BestOrGuessCase.freq',
                                               'OldScore', 'RelevantPortionOfCosim'])
    if not given_vars:
        df.to_excel("scores.xlsx")
    return df

def plot_freq(df,col):
    from matplotlib.ticker import FixedLocator, FixedFormatter
    # sort with respect to the selected column
    df_sorted = df.sort_values(by=[col])
    # use LaTeX fonts in the plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    fig, ax = plt.subplots(figsize=(9, 4))
    num_sims = len(df_sorted[col])
    sim_number = range(num_sims)
    ax.plot(sim_number,df_sorted[col]*100, color='k')
    ax.set_xlabel('Percentile of simulations')
    x_locator = FixedLocator([0, num_sims / 4, num_sims / 2, 3 * num_sims / 4, num_sims - 1])
    x_formatter = FixedFormatter([0, 25, 50, 75, 100])
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.set_xlim([0,num_sims-1])
    ax.set_ylabel(r'Percentage [\%]')
    ax.set_yticks([0,25,50,75,100])
    ax.set_ylim([-2,102])
    ax.grid(visible=True, which='major', axis='both')
    fig.tight_layout()

def plot_freq_compare(df_dict,col):
    from matplotlib.ticker import FixedLocator, FixedFormatter
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['tab:green', 'tab:olive']
    for i in range(len(df_dict)):
        adap_method = list(df_dict.keys())[i]
        df = df_dict[adap_method]
        # sort with respect to the selected column
        df_sorted = df.sort_values(by=[col])
        # use LaTeX fonts in the plot
        num_sims = len(df_sorted[col])
        sim_number = range(num_sims)
        ax.plot(sim_number,df_sorted[col]*100, color=colors[i], label='Adap ' + adap_method)
    ax.set_xlabel('Percentile of simulations')
    x_locator = FixedLocator([0, num_sims / 4, num_sims / 2, 3 * num_sims / 4, num_sims - 1])
    x_formatter = FixedFormatter([0, 25, 50, 75, 100])
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.set_xlim([0,num_sims-1])
    ax.set_ylabel(r'Percentage [\%]')
    ax.set_yticks([0,25,50,75,100])
    ax.set_ylim([-2,102])
    ax.grid(visible=True, which='major', axis='both')
    ax.legend(loc='lower right', ncol=2, fancybox=True, shadow=True, fontsize=12)
    fig.tight_layout()

def plot_score(df):
    from matplotlib.ticker import FixedLocator, FixedFormatter
    col = 'OldScore'
    # sort with respect to the selected column
    df_sorted = df.sort_values(by=[col])
    # use LaTeX fonts in the plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    fig, ax = plt.subplots(figsize=(9, 4))
    num_sims = len(df_sorted[col])
    sim_number = range(num_sims)
    ax.plot(sim_number, df_sorted[col], color='k')
    ax.set_xlabel('Percentile of simulations')
    x_locator = FixedLocator([0, num_sims / 4, num_sims / 2, 3 * num_sims / 4, num_sims - 1])
    x_formatter = FixedFormatter([0, 25, 50, 75, 100])
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.set_xlim([0,num_sims-1])
    ax.set_ylabel(r'Score')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim([0,1.0])
    ax.grid(visible=True, which='major', axis='both')
    fig.tight_layout()

if __name__ == '__main__':
    given_vars = True
    plotBool = True
    # results = resource_file_path("./datasets/parameter_random_correct_5_global")
    adap_scores = ['input','power']
    compare_score = True

    # max_applications = 10
    # apply_fun_parameter_sweep_results(results, plot_results, max_applications)
    # plt.show()
    scores_dict = {}
    for i in range(len(adap_scores)):
        if not given_vars:
            study_folder = "./datasets/parameter_random_all_estimators"
            results_dir = resource_file_path(study_folder)
            scores = apply_fun_parameter_sweep_results(results_dir, get_errors_and_score, None)
            save_vars(scores,'scores.pickle')
        else:
            # search in current directory "adaptive_cosim"
            precomputed_filename = '../../datasets/scores/scores_'+adap_scores[i]+'.pickle'
            scores = load_vars(precomputed_filename)
        df_score = export_df_score(scores, given_vars=given_vars)
        print("Score Mean: " + str(df_score["OldScore"].mean()))
        print("Score Median: " + str(df_score["OldScore"].median()))
        # print(df_score["OldScore"].mean())
        scores_dict[adap_scores[i]] = df_score

        if not compare_score:
            if plotBool:
                plot_freq(df_score, 'Score.BestCase.freq')
                plot_freq(df_score, 'Score.BestOrGuessCase.freq')
                plot_freq(df_score, 'Score.WorstCase.freq')
                plot_score(df_score)

                # parameter_list = load_vars("../../" + study_folder + "/past_parameters.pickle")
                # apply_fun_results(results_dir, X0, 0.01, 5.0, parameter_list[0], plot_results)
                #
                plt.show()
    if compare_score and plotBool:
        plot_freq_compare(scores_dict, 'Score.BestCase.freq')
        plot_freq_compare(scores_dict, 'Score.BestOrGuessCase.freq')
        plot_freq_compare(scores_dict, 'Score.WorstCase.freq')
        plt.show()
    # tf = 10.0
    # H = 0.01
    # nsamples = 3
    # # params = [0.1, 0.1, 1.0, 0.1, 0.1, 1.0, 1.0, 0.1]
    # params = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # apply_fun_results("./datasets/parameter_sweep", X0, H, tf, params, plot_results)
    # plt.show()