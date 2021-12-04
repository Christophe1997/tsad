import logging
import os
import sys

import numpy as np
import torch

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sklearn.metrics as metrics


def scan1d(arr: np.ndarray, window_size, stride=1):
    size = len(arr)
    if window_size > size:
        raise ValueError("window size must smaller than arr length")

    shape = (size - window_size) // stride + 1, window_size
    elem_size = arr.strides[-1]
    return np.lib.stride_tricks.as_strided(arr, shape, strides=(elem_size * stride, elem_size), writeable=False)


def reconstruct(arr: np.ndarray, stride=1):
    x, y = arr.shape
    res = np.zeros((x - 1) * stride + y)
    res[:y] = arr[0]
    idx = y
    for e in arr[1:]:
        res[idx: idx + stride] = e[-stride:]
        idx += stride

    return res


class CustomFormatter(logging.Formatter):

    def __init__(self, extra_field, *args, **kwargs):
        super(CustomFormatter, self).__init__(*args, **kwargs)
        self.extra_field = extra_field

    def format(self, record) -> str:
        if not hasattr(record, self.extra_field):
            setattr(record, self.extra_field, "")
        return super(CustomFormatter, self).format(record)


def get_logger(args, fp):
    res = logging.getLogger("root")
    for hadler in res.handlers[:]:
        res.removeHandler(hadler)
    fp_handler = logging.FileHandler(fp, mode='a+', encoding='utf8')
    s_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = CustomFormatter('detail', '[%(asctime)s] %(levelname)s [%(name)s] %(message)s%(detail)s')
    fp_handler.setFormatter(formatter)
    s_handler.setFormatter(formatter)
    res.addHandler(fp_handler)
    res.addHandler(s_handler)
    res.setLevel(level=args.log_level)
    return res


def create_dir(full_path):
    abs_path = os.path.abspath(full_path)
    dir_path = os.path.dirname(abs_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


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
        return "{}: {:.2f}GB/{:.2f}GB({:.2f}%)".format(device_name,
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


def stats(loss_train, loss_valid, y_, y, label, sigma=2, beta=0.1, to_html=True, html_file=None):
    (indexes,) = np.where(label == 1)
    intervals = get_intevals(indexes)
    fig = make_subplots(rows=4, cols=2,
                        specs=[[{"colspan": 2}, None],
                               [{"colspan": 2}, None],
                               [{"rowspan": 2}, {"rowspan": 2}],
                               [None, None]],
                        subplot_titles=["loss curve", "fit curve", "ROC curve", "Precision-Recall curve"])

    fig.add_trace(go.Scatter(x=np.arange(len(loss_train)), y=loss_train, mode="lines", name="train loss"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(loss_valid)), y=loss_valid, mode="lines", name="valid loss"),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=np.arange(len(y_)), y=y_, mode="lines", name="prediction",
                             line={"color": "red"}), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode="lines", name="fact",
                             line={"color": "green"}), row=2, col=1)
    min_y = np.min(y)
    max_y = np.max(y)
    for x0, x1 in intervals:
        if x0 == x1:
            fig.add_shape(go.layout.Shape(type="line", xref='x', x0=x0, y0=min_y, x1=x0, y1=max_y,
                                          line={"dash": "dash", "color": "LightSkyBlue"}), row=2, col=1)
        else:
            fig.add_shape(go.layout.Shape(type="rect", xref='x', x0=x0, y0=min_y, x1=x1, y1=max_y,
                                          line={"color": "LightSkyBlue"}, fillcolor="LightSkyBlue", opacity=0.5),
                          row=2, col=1)

    prob = np.abs(y_ - y)

    fpr, tpr, _ = metrics.roc_curve(y_true=label, y_score=prob)
    roc_auc = metrics.auc(fpr, tpr)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", fill="tozeroy", line={"color": "#FF8E04"},
                             showlegend=False), row=3, col=1)
    fig.add_shape(go.layout.Shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "blue"}),
                  row=3, col=1)
    fig.add_annotation(
        x=0.8,
        y=0.2,
        xref="x",
        yref="y",
        text=f"auc={roc_auc:.2f}",
        showarrow=False,
        align="center",
        bgcolor="#BDBDBD",
        opacity=0.8, row=3, col=1)

    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=prob)
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", fill="tozeroy", line={"color": "#FF8E04"},
                             showlegend=False), row=3, col=2)
    fig.add_shape(go.layout.Shape(type="line", x0=0, y0=1, x1=1, y1=0, line={"dash": "dash", "color": "blue"}),
                  row=3, col=2)

    fig.update_xaxes(title_text="False Positive Rate", row=3, col=1, range=[0, 1])
    fig.update_yaxes(title_text="True Positive Rate", row=3, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Recall", row=3, col=2, range=[0, 1])
    fig.update_yaxes(title_text="Precision", row=3, col=2, range=[0, 1])

    label_pred = prob > sigma
    label_pred.astype(int)
    prec, reca, f_beta, _ = metrics.precision_recall_fscore_support(label, label_pred, beta=beta,
                                                                    average="binary", zero_division=0)
    title = f"precision: {prec:.2f}, recall: {reca:.2f}, F beta score(beta={beta}): {f_beta:.2f}"

    fig.update_layout(title=title)

    if to_html:
        fig.write_html(html_file if html_file is not None else "stats.html")
    else:
        fig.show()

    return prec, reca, f_beta
