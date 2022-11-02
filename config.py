"""
Configuration file: input from the user
"""

input_directory = 'C:/Users/cace0002/AbspectroscoPY-master/'
output_directory = 'C:/Users/cace0002/AbspectroscoPY-master/'   
### DATA ASSEMBLY AND DATA QUALITY ASSESSMENT
decimal = '.'                                                   # decimal of the input file
sep = ';'                                                       # separator of the input file
header_rownr = 1                                                # header row number
possibledateheadernames = ['Date', 'Time', 'Date/Time', 'Timestamp','Measurement interval=120[sec] (Export-Aggregation disabled)'] # input possible headers of the date column 
dateparsingformat = '%Y-%m-%d %H:%M:%S'                         # format of the date 
drop_col = ['Status (Source:0)']                                # name of columns to drop; if more than one column needs to be dropped, use: (['variable_name1','variable_name2'])
header = 0                                                      # header row number
dateheadername = 'Timestamp'                                    # header of the date

fig_format = '.tiff'                                            # format of the exported figure                                       
dpi = 300                                                       # resolution of the exported figure
sample_name = 'sw'                                              # sample name 

### ATTENUATION DATA CORRECTION
nperiods = 10                                                   # number of dates to display
startwv = '700 nm'                                              # starting wavelength for the baseline correction

### DATA SMOOTHING
median_window1_min = 14                                         # window size of the median filter 1 (minutes)
median_window2_min = 30                                         # window size of the median filter 2 (minutes)
median_window3_min = 60                                         # window size of the median filter 3 (minutes)
median_window_min_selected = median_window3_min                 # specify which window size to select                    

### OUTLIER/EVENT IDENTIFICATION AND REMOVAL

sep2 =','                                                        # separator of the input event file
nevents = 2                                                      # number of typology of events
evdrop = [1, 2]                                                  # typology of event to drop (no events to drop: type None)
symbols = ['*','^']                                              # add as many symbols as typology of event

### INTERPRET THE RESULTS
# abs_slope_ratio:
S_f = 1                                                          # sampling frequency when computing absorbance spectra changes
sr_col = 'SR'                                                    # name of the column of the slope ratio dataframe to use for outliers detection

# abs_ratio:
wv1 = '220 nm'                                                   # wavelength numerator for the absorbance ratio
wv2 = '255 nm'                                                   # wavelength denominator

# abs_fit_exponential:
wl0 = 350                                                        # reference wavelength in the equation a0*exp(-S*(x-wl0))+K
startwl = 220                                                    # starting wavelength
endwl = 737.5                                                    # ending wavelength

# abs_spectral_curve
r2threshold = 0.98                                               # correlation coefficient (R2) threshold to filter the negative spectral slope results
