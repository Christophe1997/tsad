import os
import shutil

import numpy as np
import plotly.graph_objects as go
import sklearn.metrics as metrics
import torch
from plotly.subplots import make_subplots


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
    return sorted(f for f in os.listdir(root) if f.startswith("version"))[-1]


def get_last_ckpt(root, version=None):
    if version is None:
        version = get_last_version(root)
    root = os.path.join(root, version, "checkpoints")
    ckpt_path = os.path.join(root, os.listdir(root)[0])
    return ckpt_path


def copy_result(root, dest, suffixes=None, prefix="MIX", fp="test_score.pkl"):
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
            version = get_last_version(os.path.join(root, target))
            src = os.path.join(root, target, version, fp)
            dst = os.path.join(dest, s, f"{idx}_{fp}")
            print(f"{src} -> {dst}")
            if os.path.exists(src):
                shutil.copy2(src, dst)
