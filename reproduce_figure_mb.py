import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from latexify import format_axes, latexify

latexify()
# Define a list of file paths with corresponding labels
file_data = [
    ("data/llama_13B_bfloat16_64_GPUs_all_runs.xlsx", "13B"),
    ("data/llama_13B_bfloat16_8k_seq_length_128_GPUs_all_runs.xlsx", "13B_8k"),
    ("data/llama_30B_bfloat16_256_GPUs_all_runs.xlsx", "30B"),
    ("data/llama_30B_8k_seq_length_bfloat16_128_GPUs_all_runs.xlsx", "30B_8k"),
    ("data/llama_65B_bfloat16_128_GPUs_all_runs.xlsx", "65B"),
]

# Create a dictionary to store line colors for each file
line_colors = {
    "13B": "darkblue",
    "13B_8k": "darkblue",
    "30B": "darkblue",
    "30B_8k": "darkblue",
    "65B": "darkblue",
}

# Iterate through each file and generate a separate plot
for file_path, label in file_data:
    # Create an empty DataFrame to store the best entries for each micro_batch_size
    best_entries = pd.DataFrame(
        columns=[
            "Label",
            "micro_batch_size",
            "Average MFU",
            "activation_checkpointing_type",
            "tensor_parallel_size",
            "pipe_parallel_size",
        ]
    )

    # Load the Excel data into a DataFrame with header in the second row
    df = pd.read_excel(file_path, engine="openpyxl", header=1)

    # Filter the DataFrame to exclude rows with "kernel" equal to "flash_attention2 + RMS kernel"
    # df = df[df["kernel"] != "flash_attention2 + RMS kernel"]
    df = df[df["kernel"] == "flash_attention2"]

    df["Average MFU"] = pd.to_numeric(df["Average MFU"], errors="coerce")

    # Get unique micro_batch_sizes
    unique_micro_batch_sizes = df["micro_batch_size"].unique()

    # Find the row with the maximum 'Average MFU' for each micro_batch_size
    for micro_batch_size in unique_micro_batch_sizes:
        df_filtered = df[df["micro_batch_size"] == micro_batch_size]

        # Check if all results for the batch size are OOM
        if all(df_filtered["Average MFU"].isna()):
            best_entry = pd.DataFrame(
                {
                    "Label": [label],
                    "micro_batch_size": [micro_batch_size],
                    "Average MFU": [0],  # Set the value to 0 for OOM
                    "activation_checkpointing_type": [None],
                    "tensor_parallel_size": [None],
                    "pipe_parallel_size": [None],
                }
            )
            best_entries = pd.concat([best_entries, best_entry], ignore_index=True)
            pass

        else:
            best_entry = df_filtered.nlargest(1, "Average MFU")
            best_entry["Label"] = label
            best_entries = pd.concat([best_entries, best_entry], ignore_index=True)

    # Sort the best_entries DataFrame by "micro_batch_size"
    best_entries = best_entries.sort_values(by="micro_batch_size")

    # Create a plot for the current file
    fig, ax = plt.subplots(
        figsize=(4, 3.8)
    )  # Change the width (12 inches) to your desired size
    format_axes(ax)
    # plt.figure(figsize=(8, 5))  # Adjust the figure size as needed
    plt.plot(
        best_entries["micro_batch_size"],
        best_entries["Average MFU"],
        marker="o",
        label=label,
        color=line_colors[label],
    )

    # Set labels and title
    plt.xlabel("Micro-batch Size")
    plt.ylabel("Model FLOPs Utilization (\%)")
    # plt.title(f"Best Average MFU for {label}")

    # Add legend
    # plt.legend(loc="best")

    # Add annotations with (activation_checkpointing_type, tensor_parallel_size, pipe_parallel_size)
    for x, y, act_type, tensor_size, pipe_size in zip(
        best_entries["micro_batch_size"],
        best_entries["Average MFU"],
        best_entries["activation_checkpointing_type"],
        best_entries["model_parallel_size"],
        best_entries["pipe_parallel_size"],
    ):
        if act_type is None:
            annotation_text = "OOM"
        else:
            annotation_text = f"({act_type}, {tensor_size:.0f}, {pipe_size:.0f})"

        offset = 0
        if annotation_text == "(disabled, 2, 1)":
            print("offsetting")
            # Hardcode for 13B because of annotation overlap
            offset = -7

        plt.annotate(
            annotation_text,
            (x, y),
            textcoords="offset points",
            xytext=(45, 2 + offset),  # Adjust the vertical position of the annotation
            ha="center",
            fontsize=10,
            arrowprops=dict(arrowstyle="-", linestyle="--", color="black", alpha=0.8),
        )

        # Draw a dashed line connecting the annotation to the marker
        # plt.plot([x, y], [y, y + 5], linestyle="--", color="gray", alpha=0.7)
    plt.xticks([1, 2, 4, 8][: len(best_entries["micro_batch_size"])])
    # x_ticks = [0, 1, 2, 3, 4][
    #     : len(best_entries["micro_batch_size"])
    # ]  # Adjust the tick positions as needed (e.g., for 10^0, 10^1, 10^2, ...)

    # plt.xscale("log")
    ax.set_xscale("log", base=2)

    def log_x_axis_formatter(x, pos):
        # Convert the log value 'x' to its corresponding numeric value
        return f"{x:.0f}"

    def scale_y_axis_labels(value, pos):
        return (
            f"{value * 100:.0f}"  # Scale the value and format it with one decimal place
        )

    # Create a custom y-axis formatter using the scale_y_axis_labels function
    y_formatter = FuncFormatter(scale_y_axis_labels)
    ax.yaxis.set_major_formatter(y_formatter)
    # # Set the x-axis formatter for the plot
    ax.xaxis.set_major_formatter(FuncFormatter(log_x_axis_formatter))
    # # x_locator = MultipleLocator(base=1.0)  # Adjust the base as needed
    # ax.xaxis.set_major_locator(x_locator)
    # Save the plot as a PDF file
    output_filename = f"figs/mb_{label}_plot.pdf"
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
