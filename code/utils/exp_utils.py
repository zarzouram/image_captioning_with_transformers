from typing import List, Optional, Tuple, Union

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nltk.tokenize.treebank import TreebankWordDetokenizer

import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import matplotlib.colors as mcolors


def adjust_lightness(color, amount=1.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:  # noqa: E722
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def add_axis(ax, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical axis to an image plot."""
    """ref. https://stackoverflow.com/a/33505522"""
    # when I add colorbar size to the attention visualization image, the hight
    # of the color bar does not match the graph. Also, the width of color bar
    # and spacing between color bar and image changes. see:
    # https://matplotlib.org/2.0.2/mpl_toolkits/axes_grid/users/overview.html
    divider = axes_grid1.make_axes_locatable(ax)
    width = axes_grid1.axes_size.AxesY(ax, aspect=1. / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    return cax


def plot_hist(data: List[NDArray],
              fig_data: dict,
              bins: Optional[Union[List[List[float]], List[int]]] = None,
              norm_pdf: bool = False,
              count: bool = False) -> Tuple[plt.Figure, plt.Axes]:

    # print histograms with normal distribution if required
    if "figsize_factor" in fig_data:
        wf, hf = fig_data["figsize_factor"]
    else:
        wf, hf = (1.2, 1.3)

    fig_w, fig_h = plt.rcParamsDefault["figure.figsize"]
    figsize = (fig_w * wf * len(data), fig_h * hf)
    figs, axes = plt.subplots(nrows=1,
                              ncols=len(data),
                              figsize=figsize,
                              squeeze=False)
    axes_ = np.array(axes).reshape(-1)

    # get color cycles
    hist_colors = plt.get_cmap("Accent")
    line_colors = plt.get_cmap("tab10")
    text_colors = plt.get_cmap("Set1")
    # plot histogram for each data
    for i, (ax, d) in enumerate(zip(axes_, data)):
        if bins is None:
            bins = [30] * len(data)
        density, _bins, _ = ax.hist(d,
                                    bins=bins[i],
                                    density=True,
                                    alpha=0.5,
                                    color=hist_colors(i),
                                    ec=adjust_lightness(hist_colors(i)),
                                    label=fig_data["label_h"][i])

        _ = ax.set_xticks(_bins)
        _ = ax.set_xticklabels([str(round(float(b), 5)) for b in _bins],
                               rotation=90)

        # show counts on hist
        if count:
            counts, _ = np.histogram(d, _bins)
            Xs = [(e + s) / 2 for s, e in zip(_bins[:-1], _bins[1:])]
            for x, y, count in zip(Xs, density, counts):
                _ = ax.text(x,
                            y * 1.02,
                            count,
                            horizontalalignment="center",
                            rotation=45,
                            color=text_colors(i))

        # plot normal probability dist
        if norm_pdf:
            # calc normal distribution of bleu4
            d_sorted = np.sort(d)
            mu = np.mean(d)
            sig = np.std(d)
            data_norm_pdf = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(
                -np.power((d_sorted - mu) / sig, 2.) / 2)

            _ = ax.plot(d_sorted,
                        data_norm_pdf,
                        color=line_colors(i),
                        linestyle="--",
                        linewidth=2,
                        label=fig_data["label_l"][i])

        _ = ax.legend()
        _ = ax.set_xlabel(fig_data["xlabel"])
        _ = ax.set_ylabel(fig_data["ylabel"])
        y_lim = ax.get_ylim()
        _ = ax.set_ylim((y_lim[0], y_lim[1] * 1.1))

    figs.suptitle(fig_data["title"])

    return figs, axes


def get_colormap():
    # cdict["red"]:    ([threshold, r1, r2])
    # cdict["green"]:  ([threshold, g1, g2])
    # cdict["blue"]:   ([threshold, b1, b2])

    cdict = {
        "red":
        ((0.0, 1.0, 1.0), (0.2, 1.0, 1.0), (0.3, 0.0, 0.0), (0.45, 0.0, 0.0),
         (0.55, 1.0, 1.0), (0.75, 1.0, 1.0), (1.0, 1.0, 1.0)),
        "green":
        ((0.0, 1.0, 1.0), (0.2, 1.0, 1.0), (0.3, 0.5, 0.5), (0.45, 0.5, 0.5),
         (0.55, 1.0, 1.0), (0.75, 1.0, 1.0), (1.0, 0.0, 0.0)),
        "blue":
        ((0.0, 1.0, 1.0), (0.2, 1.0, 1.0), (0.3, 0.0, 0.0), (0.55, 0.0, 00.0),
         (0.55, 0.0, 0.0), (0.75, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "alpha": ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.3, 1.0, 1.0),
                  (0.8, 1.0, 1.0), (1.0, 1.0, 1.0))
    }

    return mcolors.LinearSegmentedColormap("ACustomMap", cdict)


def visualize_word_attention(image: NDArray, attn: NDArray, gt_text: str,
                             pred_text: str, word: str, widx: int,
                             save_dir: Path, bleu4_score: float, idx: int):
    figsize = (10, 12)
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    # initiate subfigures
    sfigs = fig.subfigures(3, 1, height_ratios=[5, 80, 15])
    kwarg = {
        "subplot_kw": {
            "xticklabels": [],
            "yticklabels": [],
            "xticks": [],
            "yticks": [],
            "frame_on": False
        }
    }

    # image + attention weights + colorbar axis
    ax1 = sfigs[1].subplots(**kwarg)
    ax1_cb = add_axis(ax1)
    x10, _, _, y11 = ax1.axis()

    # Predicted/Generated text axis
    ax2 = sfigs[0].subplots(**kwarg)
    x20, _, y20, y21 = ax2.axis()
    ax2.set_title(f"Predicted Caption: {bleu4_score:.5f}",
                  fontsize=15,
                  weight="semibold",
                  loc="left",
                  x=x20)

    # Refrence text
    ax3 = sfigs[2].subplots(**kwarg)
    x30, _, y30, y31 = ax3.axis()
    ax3.set_title("Ground Truth:",
                  fontsize=15,
                  weight="semibold",
                  loc="left",
                  x=x30)

    # show image and superimpose attention
    cmap = get_colormap()
    _ = ax1.imshow(image)
    attn_overlay = ax1.imshow(attn,
                              alpha=0.6,
                              interpolation="gaussian",
                              cmap=cmap)
    # show color bar
    # set color bar ticks
    minv = attn.min()
    maxv = attn.max()
    v = np.linspace(minv, maxv, 11, endpoint=True)
    vl = [f"{i:.3f}" for i in np.linspace(minv, maxv, 11, endpoint=True)]
    cbar = attn_overlay.figure.colorbar(attn_overlay, cax=ax1_cb, ticks=v)
    cbar.ax.set_yticklabels(vl)

    # add current word to image
    _ = ax1.text(x10,
                 y11,
                 word,
                 fontsize=30,
                 color="k",
                 backgroundcolor="w",
                 ha="left",
                 va="top")

    # show Refrence text
    kwargs = {"fontsize": 15, "va": "center_baseline", "wrap": True}
    _ = ax3.text(x30, (y30 + y31) / 2, gt_text, **kwargs)

    # show predicted text, plot text word by word and color the current word
    _ = ax2.text(x20, (y20 + y21) / 2, pred_text, **kwargs)

    # save plt
    score = f"{int(bleu4_score * 100000):05d}"
    save_path = save_dir / f"fig_i{idx:05d}-s{score}-w{widx+1:03d}.png"
    fig.savefig(str(save_path),
                dpi=600,
                transparent=False,
                facecolor="white",
                ext="png",
                bbox_inches="tight")
    plt.close()


def visualize_attention(image: NDArray, attns: NDArray, gt_text: str,
                        preds: List[str], bleu4_score: float, idx: int,
                        save_dir: Path):

    for widx in range(len(preds)):
        attn = attns[widx]
        pred_text = TreebankWordDetokenizer().detokenize(preds)
        word = preds[widx]
        visualize_word_attention(image, attn, gt_text, pred_text, word, widx,
                                 save_dir, bleu4_score, idx)
