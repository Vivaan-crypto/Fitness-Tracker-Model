import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

dataFrame = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

setDataFrame = dataFrame[dataFrame["set"] == 1]
plt.plot(setDataFrame["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
for label in dataFrame["label"].unique():
    subset = dataFrame[dataFrame["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

for label in dataFrame["label"].unique():
    subset = dataFrame[dataFrame["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100
# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
categoryDataFrame = (
    dataFrame.query("label == 'squat'").query("participant == 'A'").reset_index()
)

fig, ax = plt.subplots()
categoryDataFrame.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("Acceleration Y")
ax.set_xlabel("Samples")
plt.legend()
# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
participantDataFrame = (
    dataFrame.query("label == 'bench'").sort_values("participant").reset_index()
)
fig, ax = plt.subplots()
participantDataFrame.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("Acceleration Y")
ax.set_xlabel("Samples")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = "squat"
participant = "A"
allAxisDataFrame = (
    dataFrame.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)
fig, ax = plt.subplots()
allAxisDataFrame[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("Acceleration Y")
ax.set_xlabel("Samples")
plt.legend()
# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels = dataFrame["label"].unique()
participants = dataFrame["participant"].unique()

# Plots all the accelerometer data
for label in labels:
    for participant in participants:
        allAxisDataFrame = (
            dataFrame.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(allAxisDataFrame) > 0:
            fig, ax = plt.subplots()
            allAxisDataFrame[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("Acceleration Y")
            ax.set_xlabel("Samples")
            plt.title(f"{label}({participant})".title())
            plt.legend()

# Plots all the gyroscope data
for label in labels:
    for participant in participants:
        allAxisDataFrame = (
            dataFrame.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(allAxisDataFrame) > 0:
            fig, ax = plt.subplots()
            allAxisDataFrame[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyroscope Y")
            ax.set_xlabel("Samples")
            plt.title(f"{label}({participant})".title())
            plt.legend()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
label = "row"
participant = "A"
combinedDataFrame = (
    dataFrame.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combinedDataFrame[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combinedDataFrame[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].set_xlabel("samples")
# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
dataFrame = pd.read_pickle("../../data/interim/01_data_processed.pkl")

labels = dataFrame["label"].unique()
participants = dataFrame["participant"].unique()

# Plots all the accelerometer and gyroscope data in a combined graph.
# It helps in visualizing the data and will also help in training the model.
for label in labels:
    for participant in participants:
        combinedDataFrame = (
            dataFrame.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index(drop=True)
        )

        if(len(combinedDataFrame) > 0):
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combinedDataFrame[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combinedDataFrame[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
    
            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].set_xlabel("samples")
    
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()
