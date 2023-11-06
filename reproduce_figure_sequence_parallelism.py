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
    "data/seqpara/llama_13B_bfloat16_32_GPUs_sequence_parallelism.xlsx",
    "data/seqpara/llama_13B_bfloat16_8k_seq_length_64_GPUs_sequence_parallelism.xlsx",
    "data/seqpara/llama_30B_bfloat16_64_GPUs_sequence_parallelism.xlsx",
    "data/seqpara/llama_30B_8k_seq_length_bfloat16_64_GPUs_sequence_parallelism.xlsx",
    "data/seqpara/llama_65B_bfloat16_64_GPUs_sequence_parallelism.xlsx",
]

# Create a mapping for shortened file names
shortened_names = {
    "llama_13B_bfloat16_32_GPUs_sequence_parallelism": "13B",
    "llama_13B_bfloat16_8k_seq_length_64_GPUs_sequence_parallelism": "13B 8k",
    "llama_30B_bfloat16_64_GPUs_sequence_parallelism": "30B",
    "llama_30B_8k_seq_length_bfloat16_64_GPUs_sequence_parallelism": "30B 8k",
    "llama_65B_bfloat16_64_GPUs_sequence_parallelism": "65B",
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
legend_entries = ["Enabled", "Disabled"]

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

    # Filter the DataFrame for runs with sequence_parallel True and False
    df_filtered_sequence_parallel = df[(df["sequence_parallel"] == True)]
    df_filtered_not_sequence_parallel = df[(df["sequence_parallel"] == False)]

    # Find the row with the maximum 'Average MFU' for 'every_layer' and 'disabled'
    best_entry_sequence_parallel = df_filtered_sequence_parallel.nlargest(1, "Average MFU")
    best_entry_not_sequence_parallel = df_filtered_not_sequence_parallel.nlargest(1, "Average MFU")

    # Use the file name as the custom label
    custom_labels.append(f"{shortened_name}")

    # Calculate the x-position for the center of each group of bars
    x_position = x_counter + bar_width
    x_positions.extend([x_position])

    # Plot bars for 'Enabled', 'Disabled', and 'Disabled (No RMSKernel)' entries with distinct colors
    b1 = plt.bar(
        x_counter,
        best_entry_sequence_parallel["Average MFU"],
        color=colors["Disabled"],
        width=bar_width,
    )
    b2 = plt.bar(
        x_counter + bar_width,
        best_entry_not_sequence_parallel["Average MFU"],
        color=colors["Enabled"],
        width=bar_width,
    )
    b1 = b1.patches[0]
    b2 = b2.patches[0]
    # Add annotations (triples) above each bar with snake_case labels
    ANNO_SIZE = 8
    plt.annotate(
        f"({best_entry_sequence_parallel['micro_batch_size'].values[0]}, {best_entry_sequence_parallel['model_parallel_size'].values[0]}, {best_entry_sequence_parallel['pipe_parallel_size'].values[0]})",
        (b1.get_x() + b1.get_width() / 2, b1.get_height()),
        ha="center",
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        fontsize=ANNO_SIZE,
    )

    plt.annotate(
        f"({best_entry_not_sequence_parallel['micro_batch_size'].values[0]}, {best_entry_not_sequence_parallel['model_parallel_size'].values[0]}, {best_entry_not_sequence_parallel['pipe_parallel_size'].values[0]})",
        (b2.get_x() + b2.get_width() / 2, b2.get_height()),
        ha="center",
        xytext=(0, 3),  # 3 points vertical offset
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
ax.legend(legend_entries, loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, frameon=False, fancybox=True, shadow=True, fontsize=8)
# Set labels and title

format_axes(ax)
plt.xlabel("Model Type")
plt.ylabel("Model FLOPs Utilization (\%)")
plt.title("Model FLOPs Utilization with Sequence Parallelism", y=1.05)

# Display the plot
plt.tight_layout()
plt.show()
plt.savefig("figs/fig_sequence_parallelism.pdf")
