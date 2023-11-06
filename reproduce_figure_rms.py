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
RMSKERNEL = "RMSKernel (best layout)"
NO_RMSKERNEL = "No RMSKernel (best layout)"
NO_RMSKERNEL_RMS_LAYOUT = "No RMSKernel (RMS layout)"
colors = {
    RMSKERNEL: "red",
    NO_RMSKERNEL: "blue",
    NO_RMSKERNEL_RMS_LAYOUT: "green",
    # "CUDA Kernel": "green",
    # "PyTorch": "red",
}
legend_entries = [
    RMSKERNEL,
    NO_RMSKERNEL,
    NO_RMSKERNEL_RMS_LAYOUT,
]

# Counter to determine x-coordinates for bars
x_counter = 0

# Width of each bar group
bar_width = 1.2

# Create a list to store x-axis positions
x_positions = []
fig, ax = plt.subplots(
    figsize=(8, 3.8)
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
    df_flash2 = df[(df["kernel"] == "flash_attention2")]
    # df_fused = df[(df["kernel"] == "fused")]
    # df_torch = df[(df["kernel"] == "torch")]

    # Find the row with the maximum 'Average MFU' for 'every_layer' and 'disabled'
    best_entry_flash2_rms = df_flash2_rms.nlargest(1, "Average MFU")
    best_entry_flash2 = df_flash2.nlargest(1, "Average MFU")

    # equivalent_flash2_rms
    equivalent_flash2_rms = df_flash2[
        (
            df_flash2["micro_batch_size"]
            == best_entry_flash2_rms["micro_batch_size"].values[0]
        )
        & (
            df_flash2["model_parallel_size"]
            == best_entry_flash2_rms["model_parallel_size"].values[0]
        )
        & (
            df_flash2["pipe_parallel_size"]
            == best_entry_flash2_rms["pipe_parallel_size"].values[0]
        )
    ]
    # remove nan MFU values from equivalent_flash2_rms
    equivalent_flash2_rms = equivalent_flash2_rms[
        equivalent_flash2_rms["Average MFU"].notna()
    ]
    # best_entry_fused = df_fused.nlargest(1, "Average MFU")
    # best_entry_torch = df_torch.nlargest(1, "Average MFU")

    # Use the file name as the custom label
    custom_labels.append(f"{shortened_name}")
    bar_counter = 0

    # Plot bars for 'Enabled', 'Disabled', and 'Disabled (No RMSKernel)' entries with distinct colors
    rms_bar = plt.bar(
        x_counter,
        best_entry_flash2_rms["Average MFU"],
        color=colors[RMSKERNEL],
        width=bar_width,
    )
    bar_counter += 1

    bar_counter += 1
    no_rms_bar = plt.bar(
        x_counter + bar_width,
        best_entry_flash2["Average MFU"],
        color=colors[NO_RMSKERNEL],
        width=bar_width,
    )

    bar_counter += 1
    equivalent_flash2_rms_bar = plt.bar(
        x_counter + bar_width * 2,
        equivalent_flash2_rms["Average MFU"],
        color=colors[NO_RMSKERNEL_RMS_LAYOUT],
        width=bar_width,
    )
    # if len(best_entry_fused["Average MFU"]):
    #     b2 = plt.bar(
    #         x_counter + bar_counter * bar_width,
    #         best_entry_fused["Average MFU"],
    #         color=colors["CUDA Kernel"],
    #         width=bar_width,
    #     )
    #     bar_counter += 1

    # if len(best_entry_torch["Average MFU"]):
    #     b4 = plt.bar(
    #         x_counter + bar_counter * bar_width,
    #         best_entry_torch["Average MFU"],
    #         color=colors["PyTorch"],
    #         width=bar_width,
    #     )
    #     bar_counter += 1

    no_rms_bar = no_rms_bar.patches[0]
    # b2 = b2.patches[0]
    rms_bar = rms_bar.patches[0]
    equivalent_flash2_rms_bar = equivalent_flash2_rms_bar.patches[0]
    # Add annotations (triples) above each bar with snake_case labels
    ANNO_SIZE = 6
    plt.annotate(
        f"({best_entry_flash2['micro_batch_size'].values[0]}, {best_entry_flash2['model_parallel_size'].values[0]}, {best_entry_flash2['pipe_parallel_size'].values[0]})",
        (no_rms_bar.get_x() + no_rms_bar.get_width() / 2, no_rms_bar.get_height()),
        ha="center",
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        fontsize=ANNO_SIZE,
    )
    plt.annotate(
        f"({best_entry_flash2_rms['micro_batch_size'].values[0]}, {best_entry_flash2_rms['model_parallel_size'].values[0]}, {best_entry_flash2_rms['pipe_parallel_size'].values[0]})",
        (rms_bar.get_x() + rms_bar.get_width() / 2, rms_bar.get_height()),
        ha="center",
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        fontsize=ANNO_SIZE,
    )

    print(equivalent_flash2_rms)
    print(equivalent_flash2_rms_bar.get_height())
    print(
        f"({equivalent_flash2_rms['micro_batch_size'].values[0]}, {equivalent_flash2_rms['model_parallel_size'].values[0]}, {equivalent_flash2_rms['pipe_parallel_size'].values[0]})",
    )

    plt.annotate(
        f"({equivalent_flash2_rms['micro_batch_size'].values[0]}, {equivalent_flash2_rms['model_parallel_size'].values[0]}, {equivalent_flash2_rms['pipe_parallel_size'].values[0]})",
        (
            equivalent_flash2_rms_bar.get_x()
            + equivalent_flash2_rms_bar.get_width() / 2,
            equivalent_flash2_rms_bar.get_height(),
        ),
        ha="center",
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        fontsize=ANNO_SIZE,
    )

    # Calculate the x-position for the center of each group of bars
    x_position = x_counter + bar_width * (bar_counter - 1) / 2
    x_positions.extend([x_position])
    # Update the x-coordinate counter for the next group of bars
    x_counter += bar_counter + bar_width


# Reset the index of the result DataFrame
best_entries.reset_index(drop=True, inplace=True)

# Set the x-axis ticks and labels
plt.xticks(x_positions, custom_labels, rotation=45, ha="right")
# plt.legend(legend_entries, loc="upper right", frameon=False, y=1.1)
ax.legend(
    legend_entries,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=4,
    frameon=False,
    fancybox=True,
    shadow=True,
)
# Set labels and title

format_axes(ax)
plt.xlabel("Model Type")
plt.ylabel("Model FLOPs Utilization (\%)")
plt.title("Influence of the RMSNorm Kernel on Model FLOPs Utilization", y=1.05)

# Display the plot
plt.tight_layout()
plt.show()
plt.savefig("figs/fig_rms.pdf")
