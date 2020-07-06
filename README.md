# Synthetic_SonicLog_Generation
Well logs are interpreted/processed to estimate the in-situ petrophysical and geomechanical properties, which is essential for subsurface characterization. Various types of logs exist, and each provides distinct information about subsurface properties. Certain well logs, like gamma ray (GR), resistivity, density, and neutron logs, are considered as “easy-to-acquire” conventional well logs that are run in most of the wells. Other well logs, like nuclear magnetic resonance, dielectric dispersion, elemental spectroscopy, and sometimes sonic logs, are only run in limited number of wells.

Sonic travel-time logs contain critical geomechanical information for subsurface characterization around the wellbore. Often, sonic logs are required to complete the well-seismic tie workflow or geomechanical properties prediction. When sonic logs are absent in a well or an interval, a common practice is to synthesize them based on its neighboring wells that have sonic logs. This is referred to as sonic log synthesis or pseudo sonic log generation.

Compressional travel-time (DTC) and shear travel-time (DTS) logs are not acquired in all the wells drilled in a field due to financial or operational constraints. Under such circumstances, machine learning techniques can be used to predict DTC and DTS logs to improve subsurface characterization. The goal of the “SPWLA’s 1st Petrophysical Data-Driven Analytics Contest” is to develop data-driven models by processing “easy-to-acquire” conventional logs from Well #1, and use the data-driven models to generate synthetic compressional and shear travel-time logs (DTC and DTS, respectively) in Well #2. A robust data-driven model for the desired sonic-log synthesis will result in low prediction errors, which can be quantified in terms of Root Mean Squared Error by comparing the synthesized and the original DTC and DTS logs.

We have two datasets: train.csv and test.csv. We build a generalizable data-driven models using train dataset. Following that, we will deploy the newly developed data-driven models on test dataset to predict DTS and DTC logs. The data-driven model should use feature sets derived from the following 7 logs: Caliper, Neutron, Gamma Ray, Deep Resistivity, Medium Resistivity, Photo-electric factor and density. The data-driven model should synthesize two target logs: DTC and DTS logs.

1.3. Data Decription
Files

> #### train.csv All the values equals to -999 are marked as missing values.

    CAL - Caliper, unit in Inch,
    CNC - Neutron, unit in dec
    GR - Gamma Ray, unit in API
    HRD - Deep Resisitivity, unit in Ohm per meter,
    HRM - Medium Resistivity, unit in Ohm per meter,
    PE - Photo-electric Factor, unit in Barn,
    ZDEN - Density, unit in Gram per cubit meter,
    DTC - Compressional Travel-time, unit in nanosecond per foot,
    DTS - Shear Travel-time, unit in nanosecond per foot,

> #### test.csv The test data has all features that you used in the train dataset, except the two sonic curves DTC and DTS.
