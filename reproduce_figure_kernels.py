"""
Created with the friendly help of ChatGPT, after a lot of unfriendly coercion :)
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter  # Import the FuncFormatter

from latexify import format_axes, latexify

latexify()
# Define a list of file paths
file_paths = [
    "data/llama_13B_bfloat16_64_GPUs_all_runs.xlsx",
    "data/llama_13B_bfloat16_8k_seq_length_128_GPUs_all_runs.xlsx",
    "data/llama_30B_bfloat16_256_GPUs_all_runs.xlsx",
    "data/llama_30B_8k_seq_length_bfloat16_128_GPUs_all_runs.xlsx",
    "data/llama_65B_bfloat16_128_GPUs_all_runs.xlsx",
]

# Create a mapping for shortened file names
shortened_names = {
    "llama_13B_bfloat16_64_GPUs_all_runs": "13B",
    "llama_13B_bfloat16_8k_seq_length_128_GPUs_all_runs": "13B 8k",
    "llama_30B_bfloat16_256_GPUs_all_runs": "30B",
    "llama_30B_8k_seq_length_bfloat16_128_GPUs_all_runs": "30B 8k",
    "llama_65B_bfloat16_128_GPUs_all_runs": "65B",
}

# Create an empty DataFrame to store the best entries
best_entries = pd.DataFrame(
    columns=[
        "File",
        "Activation Checkpointing Type",
        "Kernel",
        "Micro Batch Size",
        "Model Parallel Size",
        "Pipe Parallel Size",
        "Average MFU",
    ]
)

# Define a list to store the custom x-axis labels
custom_labels = []

# Define distinct colors for 'Enabled', 'Disabled', and 'Disabled (No RMSKernel)'
colors = {
    "FlashAttention2 + RMS Kernel": "black",
    "FlashAttention2": "red",
    "FlashAttention1.0.8": "blue",
    "CUDA Kernel": "green",
    "PyTorch": "orange",
}
legend_entries = [
    "FlashAttention2 + RMS Kernel",
    "FlashAttention2",
    "FlashAttention1.0.8",
    "CUDA Kernel",
    "PyTorch",
]

# Counter to determine x-coordinates for bars
x_counter = 0

# Width of each bar group
bar_width = 1.42

# Create a list to store x-axis positions
x_positions = []
fig, ax = plt.subplots(
    figsize=(10, 3.8)
)  # Change the width (12 inches) to your desired size


def scale_y_axis_labels(value, pos):
    return f"{value * 100:.0f}"  # Scale the value and format it with one decimal place


# Create a custom y-axis formatter using the scale_y_axis_labels function
y_formatter = FuncFormatter(scale_y_axis_labels)

# Set the y-axis formatter for the plot
ax.yaxis.set_major_formatter(y_formatter)
# Iterate over each file
for file_path in file_paths:
    # Extract the filename without the directory path or file extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Get the shortened name from the mapping
    shortened_name = shortened_names.get(file_name, file_name)

    # Load the Excel data into a DataFrame with header in the second row
    df = pd.read_excel(file_path, engine="openpyxl", header=1)

    df["Average MFU"] = pd.to_numeric(df["Average MFU"], errors="coerce")

    # Filter the DataFrame for runs with activation_checkpointing_type 'every_layer', 'disabled', and 'kernel'
    df_flash2_rms = df[(df["kernel"] == "flash_attention2 + RMS kernel")]
    df_flash108 = df[(df["kernel"] == "flash_attentionv1.0.8")]
    df_flash2 = df[(df["kernel"] == "flash_attention2")]
    df_fused = df[(df["kernel"] == "fused")]
    df_torch = df[(df["kernel"] == "torch")]

    # Find the row with the maximum 'Average MFU' for 'every_layer' and 'disabled'
    best_entry_flash2_rms = df_flash2_rms.nlargest(1, "Average MFU")
    best_entry_flash108 = df_flash108.nlargest(1, "Average MFU")
    best_entry_flash2 = df_flash2.nlargest(1, "Average MFU")
    best_entry_fused = df_fused.nlargest(1, "Average MFU")
    best_entry_torch = df_torch.nlargest(1, "Average MFU")

    # Use the file name as the custom label
    custom_labels.append(f"{shortened_name}")

    # Plot bars for 'Enabled', 'Disabled', and 'Disabled (No RMSKernel)' entries with distinct colors
    bar_counter = 0
    for entry, label in zip(
        [
            best_entry_flash2_rms,
            best_entry_flash2,
            best_entry_flash108,
            best_entry_fused,
            best_entry_torch,
        ],
        legend_entries,
    ):
        if len(entry["Average MFU"]):
            bar = plt.bar(
                x_counter + bar_counter * bar_width,
                entry["Average MFU"],
                color=colors[label],
                width=bar_width,
            ).patches[0]
            plt.annotate(
                f"({entry['micro_batch_size'].values[0]}, {entry['model_parallel_size'].values[0]}, {entry['pipe_parallel_size'].values[0]})",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                xytext=(0, 4),  # 3 points vertical offset
                textcoords="offset points",
                fontsize=8,
            )
            bar_counter += 1

    # Calculate the x-position for the center of each group of bars
    x_position = x_counter + bar_width * (bar_counter - 1) / 2
    x_positions.extend([x_position])
    # Update the x-coordinate counter for the next group of bars
    x_counter += bar_counter + bar_width + 1


# Reset the index of the result DataFrame
best_entries.reset_index(drop=True, inplace=True)

# Set the x-axis ticks and labels
plt.xticks(x_positions, custom_labels, rotation=45, ha="right")
# plt.legend(legend_entries, loc="upper right", frameon=False, y=1.1)
ax.legend(
    legend_entries,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=5,
    frameon=False,
    fancybox=True,
    shadow=True,
    fontsize=8,
)
# Set labels and title

format_axes(ax)
plt.xlabel("Model Type")
plt.ylabel("Model FLOPs Utilization (\%)")
plt.title(
    "Influence of Optimized Implementations on Model FLOPs Utilization",
    y=1.05,
)

# Display the plot
plt.tight_layout()
plt.show()
plt.savefig("figs/fig_kernels.pdf")
