import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
singleFileAccel = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
singleFileGyro = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")
len(files)
# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
dataPath = "../../data/raw/MetaMotion\\"
f = files[0]

participant = f.split("-")[0].replace(dataPath, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

dataFrame = pd.read_csv(f) 

dataFrame["participant"] = participant
dataFrame["label"] = label
dataFrame["category"] = category


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
accelDataFrame = pd.DataFrame()
gyroDataFrame = pd.DataFrame()

accelSet = 1
gyroSet = 1

for f in files:
    participant = f.split("-")[0].replace(dataPath, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    dataFrame = pd.read_csv(f) 

    dataFrame["participant"] = participant
    dataFrame["label"] = label
    dataFrame["category"] = category
    
    if "Accelerometer" in f:
        dataFrame["set"] = accelSet
        accelSet+=1
        accelDataFrame = pd.concat([accelDataFrame, dataFrame])
    if "Gyroscope" in f:
        dataFrame["set"] = gyroSet
        gyroSet+=1
        gyroDataFrame = pd.concat([gyroDataFrame, dataFrame])

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
accelDataFrame.info()
pd.to_datetime(dataFrame["epoch (ms)"], unit = "ms")

accelDataFrame.index = pd.to_datetime(accelDataFrame["epoch (ms)"], unit = "ms")
gyroDataFrame.index = pd.to_datetime(gyroDataFrame["epoch (ms)"], unit = "ms")

del accelDataFrame["epoch (ms)"]
del accelDataFrame["time (01:00)"]
del accelDataFrame["elapsed (s)"]

del gyroDataFrame["epoch (ms)"]
del gyroDataFrame["time (01:00)"]
del gyroDataFrame["elapsed (s)"]

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


#Initializing the files and dataPaths.
files = glob("../../data/raw/MetaMotion/*.csv")
dataPath = "../../data/raw/MetaMotion\\"


#Function that cleans the dataset up.
def readDataFromFiles(files):
    #Initializing Dataframes for the two different sensors: accelerometer and gyroscope
    accelDataFrame = pd.DataFrame()
    gyroDataFrame = pd.DataFrame()

    accelSet = 1
    gyroSet = 1

    #Looping throught all the files, simplifying and classifying them.
    for f in files:
        participant = f.split("-")[0].replace(dataPath, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        dataFrame = pd.read_csv(f) 

        dataFrame["participant"] = participant
        dataFrame["label"] = label
        dataFrame["category"] = category
    
        if "Accelerometer" in f:
            dataFrame["set"] = accelSet
            accelSet+=1
            accelDataFrame = pd.concat([accelDataFrame, dataFrame])
        if "Gyroscope" in f:
            dataFrame["set"] = gyroSet
            gyroSet+=1
            gyroDataFrame = pd.concat([gyroDataFrame, dataFrame])
    
    #Changing indexes from 1-# to unix times/"epoch (ms)"
    accelDataFrame.index = pd.to_datetime(accelDataFrame["epoch (ms)"], unit = "ms")
    gyroDataFrame.index = pd.to_datetime(gyroDataFrame["epoch (ms)"], unit = "ms")

    #Deleting unnecessary columns
    del accelDataFrame["epoch (ms)"]
    del accelDataFrame["time (01:00)"]
    del accelDataFrame["elapsed (s)"]

    del gyroDataFrame["epoch (ms)"]
    del gyroDataFrame["time (01:00)"]
    del gyroDataFrame["elapsed (s)"]
    return accelDataFrame, gyroDataFrame

#Setting the two dataframes equal to the reorganized dataframes 
accelDataFrame, gyroDataFrame = readDataFromFiles(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

dataMerged = pd.concat([accelDataFrame.iloc[:, :3], gyroDataFrame], axis=1)

#Renamed columns
dataMerged.columns = [
    'acc_x', 
    'acc_y', 
    'acc_z', 
    'gyr_x', 
    'gyr_y',
    'gyr_z',
    "participant",
    "label",
    "category",
    "set"
    
]


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

#Custom sampling aggregation
sampling = {
    'acc_x': "mean", 
    'acc_y': "mean", 
    'acc_z': "mean", 
    'gyr_x': "mean", 
    'gyr_y': "mean",
    'gyr_z': "mean",
    'label': "last",
    'category': "last", 
    'participant': "last", 
    'set': "last"
}

#Resampling data    
dataMerged[:1000].resample(rule="200ms").apply(sampling)

#Split data by day
days = [g for n, g in dataMerged.groupby(pd.Grouper(freq="D"))]
dataResampled = pd.concat([dataFrame.resample(rule="200ms").apply(sampling).dropna() for dataFrame in days])

#Changed the set column from float64 to int32
dataResampled["set"] = dataResampled["set"].astype("int")
dataResampled.info()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

dataResampled.to_pickle("../../data/interim/01_data_processed.pkl")