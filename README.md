# [Linear Prediction Residual for Efficient Diagnosis of Parkinson's Disease from Gait](https://arxiv.org/abs/2107.12878#:~:text=Linear%20Prediction%20Residual%20for%20Efficient%20Diagnosis%20of%20Parkinson%27s%20Disease%20from%20Gait,-Shanmukh%20Alle%2C%20U&text=Parkinson%27s%20Disease%20(PD)%20is%20a,is%20mostly%20a%20clinical%20exercise.)
This repository contains the code for all the experiments discussed in the [paper](https://arxiv.org/abs/2107.12878#:~:text=Linear%20Prediction%20Residual%20for%20Efficient%20Diagnosis%20of%20Parkinson%27s%20Disease%20from%20Gait,-Shanmukh%20Alle%2C%20U&text=Parkinson%27s%20Disease%20(PD)%20is%20a,is%20mostly%20a%20clinical%20exercise.).

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

    ```
    pip install -r requirements.txt
    ```
- Download dataset from [Phisionet](https://physionet.org/content/gaitpdb/1.0.0/) and place it in ```dataset/raw```.
- Run the matlab script ```src/generateLPresidual.m``` to preprocess the dataset and generate LPresiduals
- View and Run Notebooks with Jupyter which can be started with the following command.

    ```
    jupyter
    ```
## Citation

If you find this work useful please cite.

```
@inproceedings{alle2021linear,
  title={Linear prediction residual for efficient diagnosis of Parkinson’s disease from gait},
  author={Alle, Shanmukh and Priyakumar, U},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={614--623},
  year={2021},
  organization={Springer}
}
```
