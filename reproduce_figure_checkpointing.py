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
colors = {"Enabled": "blue", "Disabled": "red"}
legend_entries = ["Disabled", "Enabled"]

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
    df_filtered_every_layer = df[(df["activation_checkpointing_type"] == "every_layer")]
    df_filtered_disabled = df[(df["activation_checkpointing_type"] == "disabled")]
    df_filtered_disabled_no_rms = df_filtered_disabled[
        df_filtered_disabled["kernel"] != "flash_attention2 + RMS kernel"
    ]

    # Find the row with the maximum 'Average MFU' for 'every_layer' and 'disabled'
    best_entry_every_layer = df_filtered_every_layer.nlargest(1, "Average MFU")
    best_entry_disabled = df_filtered_disabled.nlargest(1, "Average MFU")
    best_entry_disabled_no_rms = df_filtered_disabled_no_rms.nlargest(1, "Average MFU")

    # Use the file name as the custom label
    custom_labels.append(f"{shortened_name}")

    # Calculate the x-position for the center of each group of bars
    x_position = x_counter + 1.5 * bar_width
    x_positions.extend([x_position])

    # Plot bars for 'Enabled', 'Disabled', and 'Disabled (No RMSKernel)' entries with distinct colors
    # b1 = plt.bar(
    #     x_counter,
    #     best_entry_disabled["Average MFU"],
    #     color=colors["Disabled"],
    #     width=bar_width,
    # )
    b2 = plt.bar(
        x_counter + bar_width,
        best_entry_disabled_no_rms["Average MFU"],
        color=colors["Disabled"],
        # hatch="xxx",
        # fill=False,
        width=bar_width,
    )
    b3 = plt.bar(
        x_counter + 2 * bar_width,
        best_entry_every_layer["Average MFU"],
        color=colors["Enabled"],
        # hatch="///",
        # fill=False,
        width=bar_width,
    )
    # b1 = b1.patches[0]
    b2 = b2.patches[0]
    b3 = b3.patches[0]
    # Add annotations (triples) above each bar with snake_case labels
    ANNO_SIZE = 8
    # We know this failed, so let's hardcode it
    if file_name == "llama_30B_8k_seq_length_bfloat16_128_GPUs_all_runs":
        plt.text(
            b2.get_x() + b2.get_width() / 2,
            0.2,  # Adjust the vertical position as needed
            "CUDA Out of Memory",
            rotation="vertical",
            ha="center",
            va="center",
            fontsize=10,
            color="red",
        )
    plt.annotate(
        f"({best_entry_disabled_no_rms['micro_batch_size'].values[0]}, {best_entry_disabled_no_rms['model_parallel_size'].values[0]}, {best_entry_disabled_no_rms['pipe_parallel_size'].values[0]})",
        (b2.get_x() + b2.get_width() / 2, b2.get_height()),
        ha="center",
        xytext=(0, 4),  # 3 points vertical offset
        textcoords="offset points",
        fontsize=ANNO_SIZE,
    )
    plt.annotate(
        f"({best_entry_every_layer['micro_batch_size'].values[0]}, {best_entry_every_layer['model_parallel_size'].values[0]}, {best_entry_every_layer['pipe_parallel_size'].values[0]})",
        (b3.get_x() + b3.get_width() / 2, b3.get_height()),
        ha="center",
        xytext=(0, 4),  # 3 points vertical offset
        textcoords="offset points",
        fontsize=ANNO_SIZE,
    )

    # Update the x-coordinate counter for the next group of bars
    x_counter += 2 + bar_width

# Reset the index of the result DataFrame
best_entries.reset_index(drop=True, inplace=True)

# Set the x-axis ticks and labels
plt.xticks(x_positions, custom_labels, rotation=45, ha="right")
# plt.legend(legend_entries, loc="upper right", frameon=False, y=1.1)
ax.legend(
    legend_entries,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=3,
    frameon=False,
    fancybox=True,
    shadow=True,
    fontsize=8,
)
# Set labels and title

format_axes(ax)
plt.xlabel("Model Type")
plt.ylabel("Model FLOPs Utilization (\%)")
plt.title("Influence of Activation Checkpointing on Model FLOPs Utilization", y=1.05)

# Display the plot
plt.tight_layout()
plt.show()
plt.savefig("figs/fig_checkpointing.pdf")
