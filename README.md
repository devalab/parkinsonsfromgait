# Linear Prediction Residual for Efficient Diagnosis of Parkinson's Disease from Gait.
This repository contains the code for all the experiments discussed in the paper.

![Processing Pipeline](./Pipeline.jpg?raw=true "Processing Pipeline")


## Organization of the Repository
```
├── dataset
│   ├── demographics.csv
│   ├── demographics.txt
│   ├── PatientDemographics.csv
│   ├── raw
│   └── WalksDemographics.csv
├── Linear Prediction Residual for Efficient Diagnosis of Parkinson’s Disease from Gait.pdf
├── Pipeline.jpg
├── README.md
└── src
    ├── Ablation.ipynb
    ├── Comparisons
    │   ├── Baseline.ipynb
    │   ├── Batch.sh
    │   ├── Maachi et al.ipynb
    │   ├── TimeBaseline.py
    │   ├── timebaseline.txt
    │   ├── TimeMaachietal.py
    │   └── timeMaachietal.txt
    ├── DemographicsPreprocessing.ipynb
    ├── EvaluateValSplits
    │   ├── PatientLevelSplit.ipynb
    │   ├── WalkLevelSplit.ipynb
    │   └── WindowSplit.ipynb
    ├── generateLPresidual.m
    ├── Original.ipynb
    ├── timeLPresidual.sh
    ├── timeLPresidual.txt
    ├── timeOriginal.py
    ├── timeOriginal.sh
    └── timeOriginal.txt
```

## Getting Started
- Install Matlab
- Install Python dependencies

    ```pip install -r requirements.txt```
- Download dataset from [Phisionet](https://physionet.org/content/gaitpdb/1.0.0/) and place it in ```dataset/raw```.
- Run the matlab script ```src/generateLPresidual.m``` to preprocess the dataset and generate LPresiduals
- View and Run Notebooks with Jupyter which can be started with the following command.

    ```jupyter```
## Citation

If you find this work useful please cite.

```Comming soon```