import torch
import numpy as np
from collections import defaultdict

from reduced_attention_mapping import ConvAttentionMapping

np.random.seed(42)
torch.manual_seed(42)


def load_data(paths_and_weights):
    data = defaultdict(list)
    for path, weight in paths_and_weights:
        for k, v in torch.load(path, weights_only=False).items():
            if "reduced_attentions" in k:
                data[k].extend(v * weight)

    return data


train_paths_and_weights = [
    ("reduced_attentions/qwen-coder-reduced-attentions-4096.pt", 1),
    ("reduced_attentions/qwen-coder-reduced-attentions-8192.pt", 1),
    ("reduced_attentions/qwen-coder-reduced-attentions-16384.pt", 1),
    ("reduced_attentions/qwen-coder-reduced-attentions-32768.pt", 1),
    ("reduced_attentions/qwen-coder-reduced-attentions-longbench-v1.pt", 1),
]

test_paths_and_names = [
    ("reduced_attentions/qwen-coder-reduced-attentions-4096-test.pt", "RULER-4k"),
    ("reduced_attentions/qwen-coder-reduced-attentions-8192-test.pt", "RULER-8k"),
    ("reduced_attentions/qwen-coder-reduced-attentions-16384-test.pt", "RULER-16k"),
    ("reduced_attentions/qwen-coder-reduced-attentions-32768-test.pt", "RULER-32k"),
    (
        "reduced_attentions/qwen-coder-reduced-attentions-longbench-v1-test.pt",
        "LongBench-v1",
    ),
]

val_data = defaultdict(list)
for path, _ in test_paths_and_names:
    data = torch.load(path, weights_only=False)
    indices = np.random.permutation(len(data["draft_reduced_attentions"]))[:25]
    for k, v in data.items():
        val_data[k].extend([v[i] for i in indices])

    del data

train_data = load_data(train_paths_and_weights)
len(train_data["draft_reduced_attentions"])

mapping = ConvAttentionMapping(device="cuda").fit(
    **train_data,
    test_draft_reduced_attentions=val_data["draft_reduced_attentions"],
    test_full_reduced_attentions=val_data["full_reduced_attentions"],
    checkpoint_path="reduced_attention_mapping/qwen_coder_mappings/conv_default_seeded_checkpoint.pt",
)

mapping.save("reduced_attention_mapping/qwen_coder_mappings/conv_default_seeded.pt")
