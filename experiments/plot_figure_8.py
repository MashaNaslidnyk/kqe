import json
from os import path

import matplotlib.pylab as plt

from kqe.local_config import DATADIR, FIGSDIR

methods_for_exp = [
    ["esw", "esw_mu_normal", "ekqd_2", "mmdmulti", "mmd"],
    ["me", "nystrom", "ekqd_2", "mmdmulti", "mmd"],
    ["mom", "ekqd_2", "mmdlin", "mmdmulti", "mmd"],
]

xlabels = ["Dimension", "Number of samples", "Dimension"]
titles = ["Power decay", "Galaxy MNIST", "Power decay"]

method_linestyle = dict()
method_linestyle["esw"] = "o:"
method_linestyle["mmd"] = "*:"
method_linestyle["ekqd_2"] = "D-"
method_linestyle["esw_mu_normal"] = "^-"
method_linestyle["mmdmulti"] = "x-"
method_linestyle["mmdlin"] = "+-"
method_linestyle["me"] = "s--"  # square marker, dashed line
method_linestyle["nystrom"] = "v-."  # triangle_down marker, dash-dot line
method_linestyle["mom"] = "p:"  # pentagon marker, dotted line


method_color = dict()
method_color["esw"] = "blue"
method_color["mmd"] = "orange"
method_color["ekqd_2"] = "red"
method_color["esw_mu_normal"] = "green"
method_color["mmdmulti"] = "purple"
method_color["mmdlin"] = "brown"
method_color["me"] = "cyan"
method_color["nystrom"] = "magenta"
method_color["mom"] = "gray"

method_names = dict()
method_names["esw"] = "e-SW$_2$, $u=f/\|f\|$ for $f \sim (P_n+Q_n)/2$"
method_names["mmd"] = "MMD ($O(n^2)$)"
method_names["ekqd_2"] = "e-KQD$_2$"
method_names["esw_mu_normal"] = "e-SW$_2$, $u \sim Unif$"
method_names["mmdmulti"] = "MMD-Multi"
method_names["mmdlin"] = "MMD-Lin ($O(n)$)"
method_names["me"] = "ME"
method_names["nystrom"] = "Nyström"
method_names["mom"] = "MOM"

filenames_for_exp = [
    ["power_decay.json"],
    ["galaxy.json"],
    ["power_decay.json", "power_decay_w_mom.json"],
]

results = [
    {
        method: {int(k): v for k, v in method_dict.items()}
        for filename in filenames
        for method, method_dict in json.load(open(path.join(DATADIR, filename), "r"))[
            "rejection_rate"
        ].items()
    }
    for filenames in filenames_for_exp
]


if __name__ == "__main__":
    f, ax = plt.subplots(1, len(methods_for_exp), figsize=(4 * len(methods_for_exp), 5))
    FONTSIZE = 15

    for j, result in enumerate(results):
        print(j)

        methods = methods_for_exp[j]
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

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    for j in range(len(methods_for_exp)):
        pos = ax[j].get_position()
        ax[j].set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])

    needed_labels = [
        "e-SW$_2$, $u=f/\|f\|$ for $f \sim (P_n+Q_n)/2$",
        "e-SW$_2$, $u \sim Unif$",
        "e-KQD$_2$",
        "MMD ($O(n^2)$)",
        "MMD-Multi",
        "ME",
        "Nyström",
        "MOM",
        "MMD-Lin ($O(n)$)",
    ]

    needed_handles = []

    # Collect handles and labels from ax[1] and ax[2]
    handles0, labels0 = ax[0].get_legend_handles_labels()
    handles1, labels1 = ax[1].get_legend_handles_labels()
    handles2, labels2 = ax[2].get_legend_handles_labels()

    # Combine handles and labels
    all_handles = handles0 + handles1 + handles2
    all_labels = labels0 + labels1 + labels2

    for label in needed_labels:
        index = all_labels.index(label)
        needed_handles.append(all_handles[index])

    leg = ax[1].legend(
        needed_handles,
        needed_labels,
        bbox_to_anchor=(
            0.51,
            0.88,
        ),  # figure‐fraction: 0.5 mid‐width, 0.92 up from bottom
        loc="center",  # center the legend box on that point
        ncol=5,
        fontsize=FONTSIZE - 2,
        borderaxespad=0.5,
        bbox_transform=f.transFigure,  # <-- use the Figure's transform
    )

    f.savefig(path.join(FIGSDIR, "figure_8.pdf"))
    print("saved")
    plt.show()
