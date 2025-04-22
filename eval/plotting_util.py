import json
import os
from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def load_ruler_data(
    size: int,
    task: str,
    data_dir: str,
    indices: list[int] = list(range(500)),
) -> list[dict[str, Any]]:
    with open(f"{data_dir}/{size}/data/{task}/validation.jsonl") as f:
        lines = f.readlines()

    return [json.loads(lines[i]) for i in indices]


def grade_ruler_single(pred: str, outputs: list[str]) -> float:
    return 100 * sum([r.lower() in pred.lower() for r in outputs]) / len(outputs)


def get_existing_paths_and_sizes(
    prefix: str,
    names: list[str],
    input_sizes: list[int] = [4096, 8192, 16384, 32768, 65536],
    postfixes: list[str] = ["", "-v2", "-v3"],
) -> tuple[list[str], list[int]]:
    paths = []
    sizes = []
    for name in names:
        for size in input_sizes:
            for postfix in postfixes:
                filename = f"{prefix}-{size}-{name}{postfix}.json"
                if os.path.exists(filename):
                    paths.append(filename)
                    sizes.append(size)
    return paths, sizes


def parse_ruler_results(
    paths: list[str],
    sizes: list[int],
    tokenizer: Any,
    names: list[str],
    data_dir: str,
    num_questions: int = 200,
) -> list[dict[str, Any]]:
    data: dict[int, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    tasks: set[str] = set()
    for path, size in zip(paths, sizes):
        with open(path) as f:
            size_data: dict[str, dict[str, list]] = json.load(f)
            keys = {k for k in size_data.keys() if size_data[k]}
            if not tasks:
                tasks = keys
            else:
                tasks &= keys

            for task in tasks:
                for k in size_data[task].keys():
                    if k == "index":
                        if all(
                            i in data[size][task][k]
                            for i in size_data[task][k][:num_questions]
                        ):
                            continue
                        else:
                            assert not any(
                                i in data[size][task][k] for i in size_data[task][k]
                            )

                    data[size][task][k].extend(size_data[task][k])

    if len(tasks) != 13:
        print(f"using {len(tasks)}/13 tasks")

    results = []
    for name in names:
        result: dict[str, Any] = {"name": name}
        for size, d in data.items():
            task_scores = []
            output_lens = []
            for task in tasks:
                if f"{name}_ids" not in d[task]:
                    result[f"{task}-{size}"] = None
                    continue

                assert d[task]["index"] == list(range(len(d[task]["index"])))
                examples = load_ruler_data(size, task, data_dir, d[task]["index"])[
                    :num_questions
                ]
                outputs = [example["outputs"] for example in examples]
                preds = tokenizer.batch_decode(
                    [i[0] for i in d[task][f"{name}_ids"][:num_questions]]
                )

                if len(preds) != num_questions or len(examples) != num_questions:
                    print(f"{name=}, {size=}, {task=}, {len(preds)=}, {len(examples)=}")

                grades = [
                    grade_ruler_single(pred, output)
                    for output, pred in zip(outputs, preds)
                ]
                score = sum(grades) / len(grades)
                result[f"{task}-{size}"] = score
                task_scores.append(score)
                output_lens.extend([len(i[0]) for i in d[task][f"{name}_ids"]])

            if task_scores:
                result[f"all-{size}"] = sum(task_scores) / len(task_scores)
            else:
                result[f"all-{size}"] = None

            if output_lens:
                result[f"all-{size}-output-len"] = sum(output_lens) / len(output_lens)

        results.append(result)

    return results


def most_different(df: pd.DataFrame, name1: str, name2: str) -> list[str]:
    assert df.keys()[0] == "name"
    tasks = [k.split("-")[0] for k in df.keys()[1:]]
    dif = df[df["name"] == name2].values[0, 1:] - df[df["name"] == name1].values[0, 1:]
    task_dif = defaultdict(list)
    for t, d in zip(tasks, dif):
        task_dif[t].append(d)

    return sorted(set(tasks), key=lambda t: sum(task_dif[t]), reverse=True)


def plot_ruler_performance_multi_group(
    df: pd.DataFrame,
    task: str,
    name_groups: list[list[str]],
    titles: list[str],
    labels: Optional[list[str]] = None,
    width_scale: float = 6,
    height_scale: float = 4,
    save_path: Optional[str] = None,
) -> None:
    width = len(name_groups)
    rows = 1
    fig, axes = plt.subplots(
        rows, width, figsize=(width_scale * width, height_scale * rows)
    )

    if width == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    axes = [ax[:width] for ax in axes]

    markers = ["o", "*", "^", "+", "s", "D"] * 10

    lines = []
    new_labels = []

    for i, (names, title) in enumerate(zip(name_groups, titles)):
        row, col = divmod(i, width)
        ax = axes[row][col]

        for name, marker in zip(names, markers):
            name = name.replace("dynamic_attention_sinks", "das")
            name = name.replace("lookahead_sparse_prefill_snapkv", "SSA")
            x = [
                k.split("-")[1]
                for k in df.keys()
                if f"{task}-" in k and "output-len" not in k
            ]
            y = df[[k for k in df.keys() if f"{task}-" in k and "output-len" not in k]][
                df["name"] == name
            ].values[0]
            (line,) = ax.plot(x, y, label=name.replace("_", " "), marker=marker)

            if i == 0:
                lines.append(line)
                label = name
                for s1, s2 in [
                    ("_ks=15", ""),
                    ("_mcp=256", ""),
                    ("_mcp=512", ""),
                    ("_mcp=1024", ""),
                    ("_", " "),
                ]:
                    label = label.replace(s1, s2)

                new_labels.append(label)

        ax.set_title(title)
        ax.set_ylabel("RULER score (out of 100)")
        ax.set_xlabel("Input size (tokens)")
        ax.grid()

    if labels is None:
        labels = new_labels

    ncol = min(len(labels), 4)
    nrow = (len(labels) + ncol - 1) // ncol

    plt.tight_layout()

    fig.legend(
        lines,
        labels,
        loc="lower center",
        ncol=ncol,
        fontsize="small",
        bbox_to_anchor=(0.5, -0.05 * nrow),
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_ruler_performance_multi_task(
    df: pd.DataFrame,
    tasks: list[str],
    names: list[str],
    width: int = 3,
    title_prefix="",
    width_scale: float = 6,
    height_scale: float = 4,
    save_path: Optional[str] = None,
) -> None:
    num_tasks = len(tasks)
    rows = (num_tasks + width - 1) // width
    fig, axes = plt.subplots(
        rows, width, figsize=(width_scale * width, height_scale * rows)
    )

    if num_tasks == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    axes = [ax[:width] for ax in axes]

    markers = ["o", "*", "^", "+", "s", "D"] * 10

    lines = []
    labels = []

    for i, task in enumerate(tasks):
        row, col = divmod(i, width)
        ax = axes[row][col]

        for name, marker in zip(names, markers):
            name = name.replace("dynamic_attention_sinks", "das")
            name = name.replace("lookahead_sparse_prefill_snapkv", "SSA")
            x = [
                k.split("-")[1]
                for k in df.keys()
                if f"{task}-" in k and "output-len" not in k
            ]
            y = df[[k for k in df.keys() if f"{task}-" in k and "output-len" not in k]][
                df["name"] == name
            ].values[0]
            (line,) = ax.plot(x, y, label=name.replace("_", " "), marker=marker)

            if i == 0:
                lines.append(line)
                labels.append(name.replace("_ks=15", "").replace("_", " "))

        ax.set_title(title_prefix + task.replace("_", " "))
        ax.set_ylabel("RULER score (out of 100)")
        ax.set_xlabel("Input size (tokens)")
        ax.grid()

    for i in range(num_tasks, rows * width):
        row, col = divmod(i, width)
        fig.delaxes(axes[row][col])

    plt.tight_layout()

    fig.legend(
        lines,
        labels,
        loc="lower center",
        ncol=len(names) // 2,
        fontsize="small",
        bbox_to_anchor=(0.5, -0.15),
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
