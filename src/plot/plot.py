from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def _plot(**kwargs):
    path = None
    if 'path' in kwargs:
        path = kwargs['path']
    if path is None:
        plt.show()
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=480)
    plt.close()

def _title(ax, **kwargs):
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

def _xticks(**kwargs):
    rotation = None
    if 'xticks_rotation' in kwargs:
        rotation = kwargs['xticks_rotation']
    plt.xticks(rotation=rotation)

def _ylabel(ax, **kwargs):
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])

def _ylim(ax, **kwargs):
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])

# Based on: https://stackoverflow.com/a/49601444
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def simple_bar(x, y, xtick_label=None, ylim=None, path=None):
    fig, ax = plt.subplots()

    x = np.arange(len(x))  # the label locations
    y = y

    ax.bar(x, y)
    plt.xticks(rotation=90)

    if ylim:
        plt.ylim(ylim)

    if xtick_label:
        ax.set_xticks(x, xtick_label)

    plt.tight_layout()
    _plot(path=path)

def boxplot(data, **kwargs):
    fig, ax = plt.subplots()

    labels = None
    if 'labels' in kwargs:
        labels = kwargs['labels']

    plt.grid(axis='y')

    bplot = ax.boxplot(
            data,
            vert=True,
            labels=labels
            )


    _xticks(**kwargs)
    _ylabel(ax, **kwargs)
    _title(ax, **kwargs)

    plt.tight_layout()
    _plot(**kwargs)
