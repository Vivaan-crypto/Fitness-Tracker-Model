import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
from TemporalAbstraction import NumericalAbstraction

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

dataFrame = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictorColumns = list(dataFrame.columns[:6])

# Plot settings

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictorColumns:
    dataFrame[col] = dataFrame[col].interpolate()


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------


for s in dataFrame["set"].unique():
    start = dataFrame[dataFrame["set"] == s].index[0]
    stop = dataFrame[dataFrame["set"] == s].index[-1]

    duration = stop - start
    dataFrame.loc[(dataFrame["set"] == s), "duration"] = duration.seconds

durationDataFrame = dataFrame.groupby(["category"])["duration"].mean()
durationDataFrame.iloc[0] / 5
durationDataFrame.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
dataFrameLowPass = dataFrame.copy()
lowPass = LowPassFilter()

fs = 1000 / 200
cutoffFrequency = 1.3
dataFrameLowPass = lowPass.low_pass_filter(
    dataFrameLowPass, "acc_y", fs, cutoffFrequency, order=5
)

subset = dataFrameLowPass[dataFrameLowPass["set"] == 58]
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw_data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)


# Looping through all the columns and applying a low pass filter
dataFrameLowPass = dataFrame.copy()
for col in predictorColumns:
    dataFrameLowPass = lowPass.low_pass_filter(
        dataFrameLowPass, col, fs, cutoffFrequency, order=5
    )
    dataFrameLowPass[col] = dataFrameLowPass[col + "_lowpass"]
    del dataFrameLowPass[col + "_lowpass"]
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
dataFramePCA = dataFrameLowPass.copy()
PCA = PrincipalComponentAnalysis()

PCValues = PCA.determine_pc_explained_variance(dataFrame, predictorColumns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictorColumns) + 1), PCValues)
plt.xlabel("Principal Component Number")
plt.ylabel("Explained Variance")
plt.show()

dataFramePCA = PCA.apply_pca(dataFramePCA, predictorColumns, 3)

subset = dataFramePCA[dataFrameLowPass["set"] == 35]
subset[
    [
        "pca_1",
        "pca_2",
        "pca_3",
    ]
].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

dataFrameSquared = dataFramePCA.copy()

acc_r = (
    dataFrameSquared["acc_x"] ** 2
    + dataFrameSquared["acc_y"] ** 2
    + dataFrameSquared["acc_z"] ** 2
)
gyr_r = (
    dataFrameSquared["gyr_x"] ** 2
    + dataFrameSquared["gyr_y"] ** 2
    + dataFrameSquared["gyr_z"] ** 2
)
dataFrameSquared["acc_r"] = np.sqrt(acc_r)
dataFrameSquared["gyr_r"] = np.sqrt(gyr_r)

subset = dataFrameSquared[dataFrameLowPass["set"] == 14]
subset[
    [
        "acc_r",
        "gyr_r",
    ]
].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

dataFrameTemporal = dataFrameSquared.copy()
numAbs = NumericalAbstraction()

predictorColumns += ["acc_r", "gyr_r"]

windowSize = int(1000 / 200)

for col in predictorColumns:
    dataFrameTemporal = numAbs.abstract_numerical(
        dataFrameTemporal, [col], windowSize, "mean"
    )
    dataFrameTemporal = numAbs.abstract_numerical(
        dataFrameTemporal, [col], windowSize, "std"
    )


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
dataFrameFrequency = dataFrameTemporal.copy().reset_index()


FreqAps = FourierTransformation()

frameSize = int(1000 / 200)
windowSize = int(2800 / 200)
dataFrameFrequency = FreqAps.abstract_frequency(
    dataFrameFrequency, ["acc_y"], windowSize, frameSize
)
# visualize features
subset = dataFrameFrequency[dataFrameFrequency["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

dataFrameFrequencyList = []
for s in dataFrameFrequency["set"].unique():
    print(f"Applying transformation to set {s}")
    subset = (
        dataFrameFrequency[dataFrameFrequency["set"] == s].reset_index(drop=True).copy()
    )
    subset = FreqAps.abstract_frequency(subset, predictorColumns, windowSize, frameSize)
    dataFrameFrequencyList.append(subset)

dataFrameFrequency = pd.concat(dataFrameFrequencyList).set_index(
    "epoch (ms)", drop=True
)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
subset = dataFrameFrequency[dataFrameFrequency["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
    ]
].plot()

dataFrameFrequency = dataFrameFrequency.dropna()
dataFrameFrequency = dataFrameFrequency.iloc[::2]

subset = dataFrameFrequency[dataFrameFrequency["set"] == 15]
subset[
    [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
    ]
].plot()
# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
dataFrameCluster = dataFrameFrequency.copy()

clusterCol = ["acc_x", "acc_y", "acc_z"]
kVals = range(2, 10)
inertias = []

for k in kVals:
    subset = dataFrameCluster[clusterCol]
    kMeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    clusterLabel = kMeans.fit_predict(subset)
    inertias.append(kMeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(kVals, inertias)
plt.xlabel("k")
plt.ylabel("inertia")
plt.show()

kMeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = dataFrameCluster[clusterCol]
dataFrameCluster["cluster"] = kMeans.fit_predict(subset)

# Plotting(visualizing) the clusters

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in dataFrameCluster["cluster"].unique():
    subset = dataFrameCluster[dataFrameCluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# Plotting(visualizing) the labels

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in dataFrameCluster["label"].unique():
    subset = dataFrameCluster[dataFrameCluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

dataFrameCluster.to_pickle("../../data/interim/03_data_features.pkl")
