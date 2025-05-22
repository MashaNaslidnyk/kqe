import json
from os import path

import matplotlib.pylab as plt

from kqe.local_config import DATADIR, FIGSDIR

methods = [
    "mmdlin",
    "mmdmulti",
    "supkqd_2",
    "ekqd_2",
    "mmd",
    "ekqd_centered_2",
]

xlabels = ["Dimension", "Number of samples", "Number of samples", "Number of samples"]
titles = [
    "(a) Power Decay",
    "(b) Laplace v.s. Gaussian",
    "(c) Galaxy MNIST",
    "(d) CIFAR 10 v.s. CIFAR 10.1",
]

method_linestyle = dict()
method_linestyle["ekqd_centered_2"] = "o:"
method_linestyle["mmd"] = "*:"
method_linestyle["ekqd_2"] = "D-"
method_linestyle["supkqd_2"] = "^-"
method_linestyle["mmdmulti"] = "x-"
method_linestyle["mmdlin"] = "+-"

method_color = dict()
method_color["ekqd_centered_2"] = "blue"
method_color["mmd"] = "orange"
method_color["ekqd_2"] = "red"
method_color["supkqd_2"] = "green"
method_color["mmdmulti"] = "purple"
method_color["mmdlin"] = "brown"

method_names = dict()
method_names["ekqd_centered_2"] = "e-KQD$_2$-Centered ($O(n^2)$)"
method_names["mmd"] = "MMD ($O(n^2)$)"
method_names["ekqd_2"] = "e-KQD$_2$ ($O(n\log^2 n)$)"
method_names["supkqd_2"] = "sup-KQD$_2$ ($O(n\log^2 n)$)"
method_names["mmdmulti"] = "MMD-Multi ($O(n\log^2 n)$)"
method_names["mmdlin"] = "MMD-Lin ($O(n)$)"


filenames = [
    "power_decay.json",
    "laplace_vs_gaussian.json",
    "galaxy.json",
    "cifar.json",
]

results = [
    {
        method: {int(k): v for k, v in method_dict.items()}
        for method, method_dict in json.load(open(path.join(DATADIR, filename), "r"))[
            "rejection_rate"
        ].items()
    }
    for filename in filenames
]


if __name__ == "__main__":
    f, ax = plt.subplots(1, 4, figsize=(16, 4))

    FONTSIZE = 15

    for j, result in enumerate(results):
        x_axis = sorted([int(x) for x in result[methods[0]].keys()])
        x_positions = list(range(len(x_axis)))

        for method in methods:
            ax[j].plot(
                x_positions,
                [result[method][x] for x in x_axis],
                method_linestyle[method],
                label=method_names[method],
                linewidth=3,
                markersize=12,
                color=method_color[method],
                alpha=0.6,
            )
            ax[j].set_ylabel("Rejection rate", fontsize=FONTSIZE)

        ax[j].set_xlabel(xlabels[j], fontsize=FONTSIZE)
        ax[j].set_title(titles[j], fontsize=FONTSIZE)
        ax[j].set_xticks(x_positions)
        ax[j].set_xticklabels(x_axis)

    plt.tight_layout()
    for j in range(4):
        pos = ax[j].get_position()
        ax[j].set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    ax[2].legend(bbox_to_anchor=(2.3, 1.35), ncol=6, fontsize=FONTSIZE - 2)
    f.savefig(path.join(FIGSDIR, "figure_3.pdf"))
    print("saved")
    plt.show()
