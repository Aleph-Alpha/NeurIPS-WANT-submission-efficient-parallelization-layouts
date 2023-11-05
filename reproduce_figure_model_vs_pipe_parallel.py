import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from latexify import format_axes, latexify

latexify()

# Load the datasets
df1 = pd.read_excel(
    "data/llama_65B_bfloat16_128_GPUs_all_runs.xlsx"
)
df2 = pd.read_excel(
    "data/llama_30B_bfloat16_256_GPUs_all_runs.xlsx"
)
df3 = pd.read_excel(
    "data/llama_13B_bfloat16_8k_seq_length_128_GPUs_all_runs.xlsx"
)

# Function to preprocess and filter data
def preprocess(df, model: str):
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df.columns = [
        "Trial ID",
        "World Size",
        "Average Step Time",
        "Average Step Time * World size",
        "Average TFlops Megatron",
        "Average TFlops Aleph Alpha",
        "Average MFU",
        "activation_checkpointing_type",
        "kernel",
        "micro_batch_size",
        "model_parallel_size",
        "pipe_parallel_size",
    ]
    df["micro_batch_size"] = df["micro_batch_size"].astype(int)
    df["model_parallel_size"] = df["model_parallel_size"].astype(int)
    df["pipe_parallel_size"] = df["pipe_parallel_size"].astype(int)

    # Filter for FlashAttention2 without RMSNorm

    if model == "65B":
        df_filtered = df[
            (df["kernel"] == "flash_attention2 + RMS kernel")
            # & (df["pipe_parallel_size"] != 16)
            & (df["micro_batch_size"] == 1)
            & (df["activation_checkpointing_type"] == "disabled")
            & (~df["Average MFU"].isin(["OOM Error", "Other Error"]))
        ]
    else:
        df_filtered = df[
            (df["kernel"] == "flash_attention2 + RMS kernel")
            # ignore small model parallelizations for LLaMa 13B 8k seq len and LLaMA 30B
            & (df["pipe_parallel_size"] != 1)
            & (df["model_parallel_size"] != 1)
            & ~((df["pipe_parallel_size"] == 2) & (df["model_parallel_size"] == 2))
            & (df["micro_batch_size"] == 1)
            & (df["activation_checkpointing_type"] == "disabled")
            & (~df["Average MFU"].isin(["OOM Error", "Other Error"]))
        ]

    df_filtered["Average MFU"] = (
        pd.to_numeric(df_filtered["Average MFU"]) * 100
    )  # convert to percentage
    df_filtered["Model & Pipe Parallel Size"] = list(
        zip(df_filtered["model_parallel_size"], df_filtered["pipe_parallel_size"])
    )
    df_filtered = df_filtered.sort_values(by=["Model & Pipe Parallel Size"])
    return df_filtered


# Preprocess and filter data
df_filtered1 = preprocess(df1, model="65B")
df_filtered2 = preprocess(df2, model="30B")
df_filtered3 = preprocess(df3, model="13B")

# Combine the two filtered datasets into one for creating a common index
df_combined = pd.concat([df_filtered1, df_filtered2, df_filtered3])

# Create a dataframe with unique "Model & Pipe Parallel Size" tuples and corresponding index for the combined data
unique_tuples_combined = pd.DataFrame(
    df_combined["Model & Pipe Parallel Size"].unique(),
    columns=["Model & Pipe Parallel Size"],
)
unique_tuples_combined["Index"] = unique_tuples_combined.index

# Reorder unique_tuples_combined to start with the (2, 2) tuple
starting_tuple = (2, 8)
reordered_tuples = [starting_tuple] + [
    tup
    for tup in unique_tuples_combined["Model & Pipe Parallel Size"]
    if tup != starting_tuple
]
unique_tuples_combined = (
    unique_tuples_combined.set_index("Model & Pipe Parallel Size")
    .loc[reordered_tuples]
    .reset_index()
)
unique_tuples_combined["Index"] = unique_tuples_combined.index

# Update the indices in the original dataframes
df_filtered1 = pd.merge(
    df_filtered1, unique_tuples_combined, how="left", on="Model & Pipe Parallel Size"
)
df_filtered2 = pd.merge(
    df_filtered2, unique_tuples_combined, how="left", on="Model & Pipe Parallel Size"
)
df_filtered3 = pd.merge(
    df_filtered3, unique_tuples_combined, how="left", on="Model & Pipe Parallel Size"
)

# Create the plot
fig, ax = plt.subplots(figsize=(16, 6))

sns.lineplot(
    x="Index",
    y="Average MFU",
    data=df_filtered3,
    marker="D",
    markersize=10,
    linewidth=4,
    label="LLaMA 13B 8k seq len"
)
sns.lineplot(
    x="Index",
    y="Average MFU",
    data=df_filtered2,
    marker="D",
    markersize=10,
    linewidth=4,
    label="LLaMA 30B"
)
sns.lineplot(
    x="Index",
    y="Average MFU",
    data=df_filtered1,
    marker="D",
    markersize=10,
    linewidth=4,
    label="LLaMA 65B"
)
plt.xlabel("(Tensor Parallel Size, Pipeline Parallel Size)", fontsize=22)
plt.ylabel("Model FLOPs Utilization (\%)", fontsize=22)
plt.xticks(
    unique_tuples_combined["Index"],
    unique_tuples_combined["Model & Pipe Parallel Size"],
    rotation=45,
    fontsize=18,
)
plt.yticks(np.arange(35, 60, 5), fontsize=18)
ax.legend(fontsize=22)
plt.tight_layout()

format_axes(ax)

plt.savefig("figs/model_vs_pipe_parallel_ablation.pdf")
