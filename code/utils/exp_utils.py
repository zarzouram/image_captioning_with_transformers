from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def adjust_lightness(color, amount=1.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:  # noqa: E722
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_hist(data: List[NDArray],
              fig_data=dict,
              bins: Optional[Union[List[List[float]], List[int]]] = None,
              norm_pdf: bool = False,
              count: bool = False) -> Tuple[plt.Figure, plt.Axes]:

    # print histogram with normal distribution if required
    fig_w, fig_h = plt.rcParamsDefault["figure.figsize"]
    figsize = (fig_w * 1.2 * len(data), fig_h * 1.3)
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
            bins = 30
        density, _bins, _ = ax.hist(d,
                                    bins=bins[i],
                                    density=True,
                                    alpha=0.5,
                                    color=hist_colors(i),
                                    ec=adjust_lightness(hist_colors(i)),
                                    label=fig_data["label_h"][i])
        _ = ax.set_xticks(_bins)
        _ = ax.set_xticklabels(_bins, rotation=45)

        # show counts on hist
        if count is not None:
            counts, _ = np.histogram(d, _bins)
            Xs = [(e + s) / 2 for s, e in zip(_bins[:-1], _bins[1:])]
            for x, y, count in zip(Xs, density, counts):
                _ = ax.text(x,
                            y * 1.02,
                            count,
                            horizontalalignment="center",
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
        _ = ax.set_ylim((y_lim[0], y_lim[1] * 1.2))

    figs.suptitle(fig_data["title"])

    return figs, axes
