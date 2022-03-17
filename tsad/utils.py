import os
import shutil

import numpy as np
import plotly.graph_objects as go
import sklearn.metrics as metrics
import torch
from plotly.subplots import make_subplots
from math import log, floor
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from scipy.optimize import minimize

deep_saffron = '#FF9933'
air_force_blue = '#5D8AA8'


def scan(arr: np.ndarray, window_size, stride=1):
    size = arr.shape[0]
    if window_size > size:
        raise ValueError("window size must smaller than arr length")
    elif len(arr.shape) > 2:
        raise ValueError("Only support 1d, 2d arr")

    is_1d = len(arr.shape) == 1
    if is_1d:
        arr = arr.reshape((arr.shape[0], 1))

    elem_size = arr.strides[-1]
    shape = (size - window_size) // stride + 1, window_size, arr.shape[-1]
    strides = shape[-1] * stride * elem_size, shape[-1] * elem_size, elem_size
    res = np.lib.stride_tricks.as_strided(arr, shape, strides=strides, writeable=False)
    return res


def reconstruct(arr: np.ndarray, stride=1):
    x, y, n = arr.shape
    res = np.zeros([(x - 1) * stride + y, n])
    res[:y] = arr[0]
    idx = y
    for e in arr[1:]:
        res[idx: idx + stride] = e[-stride:]
        idx += stride

    return res


def dict2str(d: dict):
    return "\n".join(f"{k}: {v}" for k, v in d.items())


def get_cuda_usage():
    if not torch.cuda.is_available():
        return "Not available"
    else:
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        resolved_mem = torch.cuda.memory_reserved(device_id) / 1024 ** 3
        allocated_mem = torch.cuda.memory_allocated(device_id) / 1024 ** 3
        return "{}: {:.2f}GB/{:.2f}GB({:.2f}%)".format(
            device_name,
            allocated_mem,
            resolved_mem,
            allocated_mem / resolved_mem * 100 if resolved_mem != 0 else 0)


def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(v) for v in hidden)


def get_intevals(indexes):
    if len(indexes) < 1:
        return indexes

    res = []
    interval = [indexes[0]]
    last_idx = indexes[0]
    for i, idx in enumerate(indexes[1:]):
        if idx != indexes[i] + 1:
            interval.append(last_idx)
            res.append(interval)
            interval = [idx]
        last_idx = idx
    interval.append(last_idx)
    res.append(interval)

    return res


def normalized(data, inf=-1, sup=1):
    data_std = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return data_std * (sup - inf) + inf


def standardize(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)


def roc_curve(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", fill="tozeroy", line={"color": "#FF8E04"}))
    fig.add_shape(go.layout.Shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "blue"}))

    fig.add_annotation(
        x=1,
        y=0,
        xref="paper",
        yref="paper",
        text=f"auc={roc_auc:.2f}",
        showarrow=False,
        align="right",
        bgcolor="#BDBDBD",
        opacity=0.8)

    fig.update_layout(width=500, height=500,
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate",
                      title=f"ROC curve")
    return fig


def precision_recall_curve(y_true, y_score):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true=y_true, probas_pred=y_score, pos_label=1)
    fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", fill="tozeroy", line={"color": "#FF8E04"}))
    fig.add_shape(go.layout.Shape(type="line", x0=0, y0=1, x1=1, y1=0, line={"dash": "dash", "color": "blue"}))
    fig.update_layout(width=500, height=500,
                      xaxis_title="Recall",
                      yaxis_title="Precision",
                      title=f"Precision-Recall curve")
    return fig


def plot(y_actual, y_pred=None, intervals=None, min_y=-1, max_y=1, max_dims=6):
    x, y = y_actual.shape
    x = np.arange(x)
    y = min(y, max_dims)
    fig = make_subplots(y, 1)
    for i in range(1, y + 1):
        fig.add_trace(go.Scatter(x=x, y=y_actual[:, i - 1], name=f"actual_{i}", line={"color": "green"}), row=i, col=1)
        if y_pred is not None:
            fig.add_trace(go.Scatter(x=x, y=y_pred[:, i - 1], name=f"pred_{i}", line={"color": "red"}), row=i, col=1)

        if intervals is not None:
            for x0, x1 in intervals:
                if x0 == x1:
                    fig.add_shape(
                        go.layout.Shape(type="line", x0=x0, y0=min_y, x1=x0, y1=max_y,
                                        line={"dash": "dash", "color": "LightSkyBlue"}),
                        row=i, col=1)
                else:
                    fig.add_shape(
                        go.layout.Shape(type="rect", x0=x0, y0=min_y, x1=x1, y1=max_y,
                                        line={"color": "LightSkyBlue"}, fillcolor="LightSkyBlue",
                                        opacity=0.5),
                        row=i, col=1)

    return fig


def relative_intervals(intervals, inf=0):
    return list(filter(lambda ls: ls[0] > 0, map(lambda ls: [ls[0] - inf, ls[1] - inf], intervals)))


def get_score(wrapper, dataloader, **kwargs):
    wrapper.model.eval()
    with torch.no_grad():
        scores, y_locs, y_scales = wrapper(dataloader, **kwargs)
    return scores, y_locs, y_scales


def get_last_version(root):
    return sorted([f for f in os.listdir(root) if f.startswith("version")],
                  key=lambda s: int(s.split("_")[-1]))[-1]


def get_last_ckpt(root, version=None):
    if version is None:
        version = get_last_version(root)
    root = os.path.join(root, version, "checkpoints")
    ckpt_path = os.path.join(root, os.listdir(root)[0])
    return ckpt_path


def copy_result(root, dest, suffixes=None, prefix="MIX", fp="test_score.pkl", default_version=None):
    if not os.path.exists(dest):
        os.makedirs(dest)

    dirs = os.listdir(root)
    if suffixes is None:
        suffixes = set(e.split("_")[-1] for e in dirs)
    else:
        suffixes = set(suffixes)

    for s in suffixes:
        os.makedirs(os.path.join(dest, s), exist_ok=True)
        targets = [e for e in dirs if e.split("_")[-1] == s]
        for target in targets:
            idx = target[len(prefix) + 1:-(len(s) + 1)]
            version = get_last_version(os.path.join(root, target)) if default_version is None else default_version
            src = os.path.join(root, target, version, fp)
            dst = os.path.join(dest, s, f"{idx}_{fp}")
            if os.path.exists(src):
                print(f"{src} -> {dst}")
                shutil.copy2(src, dst)


# the below two methods come from InterFusion(https://github.com/zhhlee/InterFusion)
# adjust for neg log prob

def get_adjusted_composite_metrics(label, score):
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])

    # now get the adjust score for segment evaluation.
    fpr, tpr, _ = metrics.roc_curve(y_true=label, y_score=score, drop_intermediate=False)
    auroc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    # validate best f1
    f1 = np.max(2 * precision * recall / (precision + recall + 1e-5))
    ap = metrics.average_precision_score(y_true=label, y_score=score, average=None)
    return auroc, ap, f1, precision, recall, fpr, tpr


def get_best_f1(label, score):
    assert score.shape == label.shape
    search_set = []
    tot_anomaly = 0
    for i in range(label.shape[0]):
        tot_anomaly += (label[i] > 0.5)
    flag = 0
    cur_anomaly_len = 0
    cur_max_anomaly_score = -1e5
    for i in range(label.shape[0]):
        if label[i] > 0.5:
            # here for an anomaly
            if flag == 1:
                cur_anomaly_len += 1
                cur_max_anomaly_score = score[i] if score[i] > cur_max_anomaly_score else cur_max_anomaly_score
            else:
                flag = 1
                cur_anomaly_len = 1
                cur_max_anomaly_score = score[i]
        else:
            # here for normal points
            if flag == 1:
                flag = 0
                search_set.append((cur_max_anomaly_score, cur_anomaly_len, True))
                search_set.append((score[i], 1, False))
            else:
                search_set.append((score[i], 1, False))
    if flag == 1:
        search_set.append((cur_max_anomaly_score, cur_anomaly_len, True))
    search_set.sort(key=lambda x: x[0])
    search_set.reverse()
    best_f1_res = - 1
    threshold = 1
    P = 0
    TP = 0
    best_P = 0
    best_TP = 0
    for i in range(len(search_set)):
        P += search_set[i][1]
        if search_set[i][2]:  # for an anomaly point
            TP += search_set[i][1]
        precision = TP / (P + 1e-5)
        recall = TP / (tot_anomaly + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        if f1 > best_f1_res:
            best_f1_res = f1
            threshold = search_set[i][0]
            best_P = P
            best_TP = TP

    return best_f1_res, threshold


class SPOT:
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)

    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user

    extreme_quantile : float
        current threshold (bound between normal and abnormal events)

    data : numpy.array
        stream

    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)

    init_threshold : float
        initial threshold computed during the calibration step

    peaks : numpy.array
        array of peaks (excesses above the initial threshold)

    n : int
        number of observed values

    Nt : int
        number of observed peaks
    """

    def __init__(self, q=1e-4):
        """
        Constructor

	    Parameters
	    ----------
	    q
	        Detection level (risk)

	    Returns
	    ----------
    	SPOT object
        """
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t extreme quantile : %s\n' % self.extreme_quantile
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        """
        Import data to SPOT object

        Parameters
	    ----------
	    init_data : list, numpy.array or pandas.Series
		    initial batch to calibrate the algorithm

        data : numpy.array
		    data for the run (list, np.array or pd.series)

        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data

        Parameters
	    ----------
	    data : list, numpy.array, pandas.Series
		    data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        """
        Run the calibration (initialization) step

        Parameters
	    ----------
        level : float
            (default 0.98) Probability associated with the initial threshold t
	    verbose : bool
		    (default = True) If True, gives details about the batch initialization
        verbose: bool
            (default True) If True, prints log
        min_extrema bool
            (default False) If True, find min extrema instead of max extrema
        """
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level

        level = level - floor(level)

        n_init = self.init_data.size

        S = np.sort(self.init_data)  # we sort X to get the empirical quantile
        self.init_threshold = S[int(level * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
		    scalar function
        jac : function
            first order derivative of the function
        bounds : tuple
            (min,max) interval for the roots search
        npoints : int
            maximum number of roots to output
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval

        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        Y : numpy.array
		    observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)

        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
		    numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q

        Parameters
        ----------
        gamma : float
            GPD parameter
        sigma : float
            GPD parameter

        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True, dynamic=True, verbose=False):
        """
        Run SPOT on the stream

        Parameters
        ----------
        with_alarm : bool
            (default = True) If False, SPOT will adapt the threshold assuming \
            there is no abnormal values


        Returns
        ----------
        dict
            keys : 'thresholds' and 'alarms'

            'thresholds' contains the extreme quantiles and 'alarms' contains \
            the indexes of the values which have triggered alarms

        """
        if self.n > self.init_data.size:
            print('Warning : the algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        # list of the thresholds
        th = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size), disable=not verbose):

            if not dynamic:
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                # If the observed value exceeds the current threshold (alarm case)
                if self.data[i] > self.extreme_quantile:
                    # if we want to alarm, we put it in the alarm list
                    if with_alarm:
                        alarm.append(i)
                    # otherwise we add it in the peaks
                    else:
                        self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                        self.Nt += 1
                        self.n += 1
                        # and we update the thresholds

                        g, s, l = self._grimshaw()
                        self.extreme_quantile = self._quantile(g, s)

                # case where the value exceeds the initial threshold but not the alarm ones
                elif self.data[i] > self.init_threshold:
                    # we add it in the peaks
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.Nt += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                else:
                    self.n += 1

            th.append(self.extreme_quantile)  # thresholds record

        return {'thresholds': th, 'alarms': alarm}

    def plot(self, run_results, with_alarm=True):
        """
        Plot the results of given by the run

        Parameters
        ----------
        run_results : dict
            results given by the 'run' method
        with_alarm : bool
		    (default = True) If True, alarms are plotted.

        Returns
        ----------
        list
            list of the plots

        """
        x = range(self.data.size)
        K = run_results.keys()

        ts_fig, = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if 'thresholds' in K:
            th = run_results['thresholds']
            th_fig, = plt.plot(x, th, color=deep_saffron, lw=2, ls='dashed')
            fig.append(th_fig)

        if with_alarm and ('alarms' in K):
            alarm = run_results['alarms']
            al_fig = plt.scatter(alarm, self.data[alarm], color='red')
            fig.append(al_fig)

        plt.xlim((0, self.data.size))

        return fig
