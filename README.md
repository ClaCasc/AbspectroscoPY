# AbspectroscoPY

### CONTENTS

AbspectroscoPY ("Absorbance spectroscopic analysis in Python") is a Python toolbox for processing time-series datasets collected by in-situ spectrophotometers. The toolbox addresses some of the main challenges in data preprocessing by handling duplicates, systematic time shifts, baseline corrections and outliers. It contains automated functions to compute a range of spectral metrics for the time-series data, including absorbance ratios, exponential fits, slope ratios and spectral curves. 

It covers the following aspects:

I. IMPORT RAW DATA FILES

II. PREPROCESS THE DATASET

A) data type conversion
B) data quality assessment
C) time axis shifting
D) attenuation data correction
E) data smoothing

III. EXPLORE THE DATASET

A) visualisation of data distribution
B) outlier/event identification and removal

IV. INTERPRET THE RESULTS

A) absorbance ratios
B) absorbance spectra changes


### FILES

The repository includes the following files:

I. TOOLBOX ("abspectroscopy_functions.py")

II. USER CONFIGURATION FILE ("config.py") The file provides an example of configuration settings for the variables used in the functions that the user can modify.

III. IPYNB FILES. These are files that allow the user to run the functions in the toolbox as a workflow ("main_workflow.ipynb") or individually. 

IV. CSV FILES. They represent an example dataset that can be used to test and understand the functions. In detail, the files included in the folder "data_scan_fp" are attenuation data from a spectro::lyser sensor (s::can Messtechnik GmbH) measuring every 2 minutes (200-750 nm) in Lake Neden, which is the surface water source for VIVAB’s Kvarnagården drinking water treatment plant in western Sweden. The files included in the folder "other_data" are an Excel file including known events with their timing and typology and its corresponding csv file.


### CITATION

To cite AbspectroscoPY use:

Cascone et al. (2022). AbspectroscoPY, a Python toolbox for absorbance-based sensor data in water quality monitoring. Environmental Science: Water Research & Technology. DOI	https://doi.org/10.1039/D1EW00416F

### REFERENCES

A. Bricaud, A. Morel and L. Prieur, Absorption by dissolved organic matter of the sea (yellow substance) in the UV and visible domains, Limnology and Oceanography, 1981, 26, 43-53.

J. R. Helms, A. Stubbins, J. D. Ritchie, E. C. Minor, D. J. Kieber and K. Mopper, Absorption spectral slopes and slope ratios as indicators of molecular weight, source, and photobleaching of chromophoric dissolved organic matter, Limnology and Oceanography, 2008, 53, 955-969.

S. A. Loiselle, L. Bracchini, A. M. Dattilo, M. Ricci, A. Tognazzi, A. Cózar and C. Rossi, Optical characterization of chromophoric dissolved organic matter using wavelength distribution of absorption spectral slopes, Limnology and Oceanography, 2009, 54, 590-597.

P. Massicotte and S. Markager, Using a Gaussian decomposition approach to model absorption spectra of chromophoric dissolved organic matter, Marine Chemistry, 2016, 180, 24-32.

R. A. Müller, D. N. Kothawala, E. Podgrajsek, E. Sahlée, B. Koehler, L. J. Tranvik and G. A. Weyhenmeyer, Hourly, daily, and seasonal variability in the absorption spectra of chromophoric dissolved organic matter in a eutrophic, humic lake, Journal of Geophysical Research: Biogeosciences, 2014, 119, 1985-1998.
