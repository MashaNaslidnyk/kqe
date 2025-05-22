import json
from os import path

import matplotlib.pylab as plt

from kqe.local_config import DATADIR, FIGSDIR

methods_for_exp = [
    ["ekqd_2", "ekqd_tr", "ekqd_rtr", "mmd"],
    ["ekqd_2", "ekqd_tr", "ekqd_rtr", "mmd"],
    ["ekqd_2", "ekqd_2_normal", "ekqd_2_uniform", "mmd"],
]

xlabels = ["Number of samples", "Number of samples", "Number of samples"]
titles = ["Galaxy MNIST", "CIFAR 10 v.s. CIFAR 10.1", "CIFAR 10 v.s. CIFAR 10.1"]

method_linestyle = dict()
method_linestyle["ekqd_tr"] = "x-"
method_linestyle["mmd"] = "*:"
method_linestyle["ekqd_2"] = "D-"
method_linestyle["ekqd_rtr"] = "v-"
method_linestyle["ekqd_2_normal"] = "o:"
method_linestyle["ekqd_2_uniform"] = "^-"

method_color = dict()
method_color["ekqd_tr"] = "blue"
method_color["mmd"] = "orange"
method_color["ekqd_2"] = "red"
method_color["ekqd_rtr"] = "brown"
method_color["ekqd_2_normal"] = "purple"
method_color["ekqd_2_uniform"] = "green"

method_names = dict()
method_names["ekqd_tr"] = "e-KQD$_2$-/\ ($O(n\log^2 n)$)"
method_names["mmd"] = "MMD ($O(n^2)$)"
method_names["ekqd_2"] = "e-KQD$_2$ ($O(n\log^2 n)$)"
method_names["ekqd_rtr"] = "e-KQD$_2$-\/ ($O(n\log^2 n)$)"
method_names["ekqd_2_normal"] = (
    "e-KQD$_2$, $\mu=($IQR$ / 1.349) \\times N(0, 1)$ ($O(n\log^2 n)$)"
)
method_names["ekqd_2_uniform"] = (
    "e-KQD$_2$, $\mu=$IQR$ \\times U[0, 1]$ ($O(n\log^2 n)$)"
)


filenames = ["galaxy.json", "cifar.json", "cifar.json"]

results = [
    {
        method: {int(k): v for k, v in method_dict.items()}
        for method, method_dict in json.load(open(path.join(DATADIR, filename), "r"))[
            "rejection_rate"
        ].items()
    }
    for methods, filename in zip(methods_for_exp, filenames)
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
        "e-KQD$_2$ ($O(n\log^2 n)$)",
        "MMD ($O(n^2)$)",
        "e-KQD$_2$-/\ ($O(n\log^2 n)$)",
        "e-KQD$_2$-\/ ($O(n\log^2 n)$)",
        "e-KQD$_2$, $\mu=($IQR$ / 1.349) \\times N(0, 1)$ ($O(n\log^2 n)$)",
        "e-KQD$_2$, $\mu=$IQR$ \\times U[0, 1]$ ($O(n\log^2 n)$)",
    ]

    needed_handles = []

    # Collect handles and labels from ax[1] and ax[2]
    handles1, labels1 = ax[1].get_legend_handles_labels()
    handles2, labels2 = ax[2].get_legend_handles_labels()

    # Combine handles and labels
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

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
        ncol=3,
        fontsize=FONTSIZE - 2,
        borderaxespad=0.5,
        bbox_transform=f.transFigure,  # <-- use the Figure's transform
    )

    f.savefig(path.join(FIGSDIR, "figure_7.pdf"))
    print("saved")
    plt.show()
