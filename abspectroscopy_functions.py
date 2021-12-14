"""
Collection of functions included in the AbspectroscoPY toolbox. 
"""
# Import packages and user configuration 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from datetime import datetime
from datetime import datetime as dt
from datetime import timedelta
import seaborn as sns
from scipy import interpolate
from sklearn.linear_model import LinearRegression
import scipy as sp 
import statistics 
from statistics import median
from scipy.optimize import curve_fit
from pylab import *
import glob

from config import *

# Functions included in the AbspectroscoPY toolbox

def get_files_list(filelist, 
                   filepattern = '*.csv'    # format of the files to import
                   ):
    '''
    function to get the list of files with a specific pattern including or less the path
    :argument indata: path where to search
    :argument filepattern: the filepattern string to match when building the lists 
    :return: one list with the full path to the files and one without path
    '''
    import os, fnmatch
    listoffileswithpath = []
    listoffilesnopath = []
    for dirname, dirnames, filenames in os.walk(filelist):
        for filename in fnmatch.filter(filenames, filepattern):
            listoffilesnopath.append(filename)                   
            filepathandname = os.path.join(dirname, filename)
            listoffileswithpath.append(filepathandname)         
    return (listoffileswithpath, listoffilesnopath)

def remove_parentheses(filelist, rownr, removeparentheses):
    '''
    function to remove parentheses
    '''
    import re
    filelist = open(filelist, 'r')
    i = 0
    newline=None
    for line in filelist:
        i = i + 1
        if (rownr == i):
            newline = line.replace(';', '\t')  
            if removeparentheses:
                newline = re.sub(r'\([^)]*\)', '', newline) 
            break            
    filelist.close()
    return newline

def guess_date_column(listoffiles, list_possibledates, rownr):
    import os
    '''
    function to determine the name of the date column using a list of possible date column names

    :argument listoffiles: A file with headers. Note, only the first file in the list will be used for guessing
    :argument list_possibledates: A list of probable date header names
    :argument rownr: The rownr containing the headers
    :return: name of a probable column containing dates
    '''
    indatafile=listoffiles[0]
    headerrow = remove_parentheses(indatafile, rownr, removeparentheses= False)
    datecolname='' 
    for dateguess in list_possibledates:
        if dateguess in headerrow:
            if len(dateguess)>len(datecolname): 
                datecolname = dateguess
                print("Guessed date column name: "+ datecolname)
    if len(datecolname)<1:        
        print("\nFailed guessing dateheader")
    return datecolname

def dateparse(x):
    '''
       function to convert a string to datetime using strptime () function
    '''
    parsed = pd.datetime.strptime(x, dateparsingformat)
    return parsed

def abs_read(listoffileswithpath, 
             listoffilesnopath,
             header_rownr,                          
             dateheadername,
             drop_col):  
    '''
    function to import a list of attenuation data files as function of time
    :argument listoffileswithpath: list of files including path (output of the function "get_files_list")
    :argument listoffilesnopath: list of files without path (output of the function "get_files_list")
    :argument header_rownr: header row number
    :argument dateheadername: name of the date column (output of the function "guess_date_column")
    :argument drop_col: drop useless columns   
    :return: dataframe with the attenuation data as function of time
    '''        
    df= pd.DataFrame() 
    i=0
    totfiles = len(listoffileswithpath)
    for file, fileshort in zip(listoffileswithpath, listoffilesnopath):
        i=i+1
        if True:
            infile_csv = file
            print("Processing : "+str(i)+"/"+str(totfiles)+" "+ file) 
            endf = pd.read_csv(filepath_or_buffer=infile_csv, sep=sep, header=header_rownr, index_col=1, 
                                decimal=decimal, low_memory=False , parse_dates=[dateheadername], 
                                date_parser=dateparse)
            endf.reset_index(level=0, inplace=True)
            df = df.append(endf, ignore_index = True, sort=False)
    df = df.set_index(dateheadername)                # set the date as index
    df = df.drop(drop_col, axis=1)                   # drop useless columns
    df_out = df.copy()
    df_out.index = df_out.index.rename('Timestamp')  # rename the index column as "Timestamp"
    return(df_out)


def convert2dtype(df_in, 
                  dateheadername):
    '''
    function to convert one or more categories of values to a different one (in this example, the column "Timestamp" 
    will be converted to the format datetime, while the other columns will be converted to floating-point numbers)
    :argument df_in: dataframe in input
    :argument dateheadername: name of the date column
    :return: dataframe with converted categories of values
    '''  
    print('Categories before conversion:', '\n', df_in.dtypes, '\n', '\n', ' Converting:')
    for i,col in enumerate(df_in.columns):                             # iterating the columns
        ifskip = True
        if col == dateheadername:
            df_in[[col]] = df_in[[col]].apply(pd.to_datetime)    
        elif col != dateheadername: 
            if df_in[col].dtype == 'object': 
                df_in[[col]] = df_in[[col]].apply(pd.to_numeric)       # convert all columns of DataFrame
                ifconv = True
    #        elif df_in[col].dtype == 'other category to convert':
    #            df_in[[col]] = df_in[[col]].apply(pd.to_numeric)      # pd.to_(category to which convert the current category)
    #            ifconv = True
            else:
                ifskip = False
        else:
            ifskip = False

        if ifskip:
            print("%5i: Converted to desired category! (%s)" %(i,col))  # python-output-formatting
        else:
            print("%5i: Skip!               (%s)" %(i,col))
    print('\n', 'Categories after conversion:', '\n', df_in.dtypes)
    df_out = df_in.copy()    
    return (df_out)


def nan_check(df_in, 
              dateheadername):
    '''
    function to quantify missing data per column and per row in percentage
    :argument df_in: dataframe in input
    :argument dateheadername: name of the date column
    :return: two dataframes with percentages of missing data per column and row
    '''    
    df_in = df_in.reset_index()
    nan_col = df_in.isnull().sum()           # check missing data per column
    rownr = len(df_in)  
    df_out1 = df_in.isnull().sum()/rownr*100 # check missing data per column in percentage
    nan_row = pd.DataFrame(columns=[dateheadername,'missing data'])
    colnr = len(df_in.columns)    
    for i in range(rownr):                   # check missing data per row
        nan_per_row = pd.DataFrame([[df_in[dateheadername][i], df_in.iloc[i].isnull().sum()]], columns=[dateheadername,'missing data'])
        nan_row = nan_row.append(nan_per_row)     
    nan_row.set_index(dateheadername, inplace = True)
    df_out2 = nan_row/colnr*100
    return(df_out1, df_out2)


def makeaplot(df_in, 
              output_dir,
              col_sel, 
              timestart, 
              timeend,
              label_name,
              title,
              dateformat = None, 
              locator = None, 
              xlabel = None, 
              ylabel = None, 
              yminlim = None, 
              ymaxlim = None, 
              ymajlocator = None, 
              yminlocator = None):
    '''
    function to get an even plot structure
    :argument df_in: dataframe in input
    :argument output_dir: directory where storing the results
    :argument col_sel: selected wavelength column
    :argument timestart, timeend: starting and ending date
    :argument label_name: sample name
    :argument title: figure title
    :argument dateformat: format of the dates to display
    :argument locator: format the date axis automatically, per day, etc. (e.g. "mdates.AutoDateLocator()"/ "mdates.DayLocator()") 
    :argument xlabel, ylabel: x- and y-axes label
    :argument yminlim, ymaxlim: y-axis limits
    :argument yminlocator, ymajlocator: y-axis major and minor ticks 
    :return: the absorbance spectra plot for different dates    
    '''   
    factor = .5
    plot_width = 25*factor                                                           # define the plot dimensions in inches (width and height)
    plot_height = 13*factor
    df_in[col_sel].plot(style = ['o'],#,'+','s','v','p'],                            # the symbol of the markers  
                        ms = 3,                                                      # the size of the markers
                        label = label_name                                           # the label in the legend
                        ) 
    if dateformat is None:
        dateformat = '%y-%m-%d'
    if locator is None:
        locator = mdates.AutoDateLocator()
    if xlabel is None:
        xlabel = 'Date [yy-mm-dd]'
    if  ylabel is None:
        ylabel = 'Absorbance' + '$_{' + col_sel + '}$' #+ '[abs m$^{-1}$]'        
    if yminlim is None:
        ymin = df_in[col_sel].min()
        yminlim = ymin - 0.05*abs(ymin)
    if ymaxlim is None:
        ymax = df_in[col_sel].max()
        ymaxlim = ymax + 0.05*abs(ymax) 
    if ymajlocator is None:
        ymajlocator = round((ymaxlim-yminlim)/3,4) 
    if yminlocator is None:
        yminlocator = ymajlocator/3
    
    plt.style.use ('tableau-colorblind10')                                            # use colors colorblind friendly
    plt.xlabel(xlabel)                                                                # label the x-axis 
    plt.ylabel(ylabel)                                                                # label the y-axis 
    plt.rc('font', family='Arial')                                                    # define the font
    plt.rcParams['axes.labelsize'] = 20                                               # define the label size
    #plt.rcParams['axes.labelweight'] = 'bold'                                        # define the label weight
    plt.gca().tick_params(axis = 'both', which = 'major', length = 5, labelsize = 18) # define the label size of the labels on the axes-ticks (major and minor) and length and color of ticks 
    plt.gca().tick_params(which = 'minor', length = 4, colors = 'k')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(dateformat))             #define the format of the date
    plt.gca().xaxis.set_major_locator(locator)                                        # e.g. "AutoDateLocator()"/ "DayLocator()"
    #plt.minorticks_on()                                                              # turn on the minor ticks on both the axes
    plt.gca().set_xlim(timestart,timeend)                                             # define the x-axis range 
    plt.gca().set_ylim(yminlim,ymaxlim)                                               # define the y-axis range
    plt.gca().yaxis.set_major_locator(MultipleLocator(ymajlocator))                   # define the y-axes major and minor ticks 
    plt.gca().yaxis.set_minor_locator(MultipleLocator(yminlocator))                     

    #plt.grid(which='major', linestyle=':', linewidth='1', color='gray')              # define the major grid
    #plt.grid(which='minor', linestyle=':', axis = 'x', linewidth='1', color='gray')  # define the minor grid

    plt.legend(loc='best', fontsize = 17, markerscale = 2)                            # define the legend 
                                                                                      
    plt.rcParams['figure.figsize'] = (plot_width, plot_height)                        # define the size of the figure
    plt.tight_layout() 
    fig = plt.gca().get_figure()                                                      # return the figure 
    fig.savefig(output_dir + title + str(label_name) + fig_format, dpi = dpi)         # save the figure to the desired format and resolution 

def dup_check(df_in,
             dateheadername):
    '''
    function to check and plot duplicates
    :argument df_in: dataframe in input
    :argument dateheadername: name of the date column
    :return: two dataframes with duplicates by dateheadername and by all columns
    '''    
    df_out1 = df_in[df_in.index.duplicated(keep=False)] # check for duplicates by dateheadername     
    df_out2 = df_in[df_in.duplicated()]                 # check for duplicates by all the columns and report them once 
    return(df_out1, df_out2)    

def tshift_dst(df_in, 
               dateheadername,
               nsamples_per_hour):
    '''
    function to shift the dataset in time one hour forward when the Daylight Saving Time ends    
    :argument df_in: dataframe in input
    :argument dateheadername: name of the date column
    :argument nsamples_per_hour: number of samples per hour
    :return: the dataframe shifted according to Daylight saving time
    
    
    '''
    df_in = df_in.reset_index()
    df_for_shift = df_in.copy()    
    df_for_shift['Time between samples'] = (df_for_shift[dateheadername] -                 # compute the time between samples
                                           df_for_shift[dateheadername].shift(1)).astype('timedelta64[m]')
    dst = df_for_shift[df_for_shift.duplicated(subset=[dateheadername], keep=False)].index #get duplicates by headername

    # Identify where to start shifting (i.e. index of the row where the first duplicate appears):
    n = 2*nsamples_per_hour
    dst_chunks = [dst[i * n:(i + 1) * n] for i in range((len(dst) + n - 1) // n )]         # divide the duplicates by headername in chunks
    shift_start = []
    for i,chunk in enumerate(dst_chunks):
        idxs = list(sorted(chunk))
        shift_start.append(idxs[nsamples_per_hour])

    # Identify where to end shifting (i.e. the time delta is over an hour):
    df_end = df_for_shift.index[df_for_shift['Time between samples'] == 63]
    # To check where the time gap is different from 3 min or bigger than 63:
    #df_for_shift.loc[df_for_shift['Time between samples'] != 3]
    #df_for_shift.loc[df_for_shift['Time between samples'] > 63]                            # gaps might be due to stop data acquisition for sensor cleaning
    shift_end = []
    for i in df_end:
        shift_end.append(i-1)

    # Shift:
    if shift_end[0] < shift_start[0]:                                                       # there is a jump first -> shift everything before
        shift_start.append(0)                                                               # else there is a duplicate first -> everything is good
    if shift_end[-1] < shift_start[-1]:                                                     # the last anomaly is a duplicate -> shift all the way to the end 
        shift_end.append(df_for_shift.index[-1])                                            # the last anomaly is a jump -> everything is good   
    timeshift = pd.Timedelta('1 hours') 
    df_out = df_for_shift.copy()
    for i,(start,end) in enumerate(zip(shift_start,shift_end)):
        print('Step',i+1,'\n\tstart:',start,'\n\tend:  ',end)
        print('Shift interval:  ',df_out.loc[start,dateheadername],'to',df_out.loc[end,dateheadername])  # choose only those rows where the index is in the range       
        shiftslice = (df_out[dateheadername].index >= start) & (df_out[dateheadername].index <= end)
        df_out.loc[shiftslice, dateheadername] = df_out.loc[shiftslice,dateheadername] + timeshift
        print('Shifting',sum(shiftslice),'rows')
    df_out.reset_index(inplace=True,drop=True)                                              # reset the index to get a continuous index
    df_out.set_index([dateheadername], inplace=True, drop=False)                            # set the date as index
    df_out.sort_index(axis = 0, inplace=True)                                               # sort by increasing index
    df_out = df_out.drop([dateheadername], axis=1)                                          # drop the column that now is also in index
    return(df_out)  


def abs_pathcor(df_in, 
                path_length):
    '''
    function to correct the attenuation data according to the path length
    :argument df_in: dataframe in input
    :argument path_length: path length of the window of the sensor
    :return: the dataframe of attenuation data corrected by the path length of the window of the sensor    
    '''
    before_correction = df_in['255 nm'][0] # attenuation value read by the sensor for the first date before path length correction 
    abs_scale = path_length 
    nrcol = len(df_in.columns)
    df_out = df_in.copy()
    for i in range(0,nrcol):
        df_out.iloc[:,i] = df_out.iloc[:,i]/abs_scale
    pd.options.display.max_rows = 15
    after_correction = df_out['255 nm'][0] # attenuation value read by the sensor for the first date after path length correction 
    
    # Compare the attenuation value read by the sensor before and after path length correction for a specific date to the value 
    # obtained in the laboratory (consider that, if analysing filtered samples in the laboratory, part of the difference in the 
    # absolute value is due to the fact that the sensor measures instead unfiltered water: the sensor data should be higher than
    # the laboratory data):
    return('before correction:', before_correction, 'after correction:', after_correction, 'dataframe after correction:', df_out) 


def makeabsplot(df_in,
                output_dir,
                dateparsingformat,
                nperiods,
                label_name,
                xlabel = None, 
                ylabel = None, 
                xminlim = None,
                xmaxlim = None,   
                yminlim = None, 
                ymaxlim = None, 
                xminlocator = None,
                xmajlocator = None,
                ymajlocator = None, 
                yminlocator = None):
    
    '''
    function to get the plot of the absorbance spectra for different dates
    :argument df_in: dataframe in input
    :argument output_dir: directory where storing the results
    :argument dateparsingformat: format of the dates 
    :argument nperiods: number of dates to display
    :argument label_name: sample name
    :argument xlabel, ylabel: x- and y-axes label
    :argument xminlim, xmaxlim, yminlim, ymaxlim: x- and y-axes limits
    :argument xminlocator, xmajlocator, yminlocator, ymajlocator: x- and y-axes major and minor ticks 
    :return: the absorbance spectra plot for different dates    
    '''   
    factor = .5                                                                      # define the plot dimensions in inches (width and height)
    plot_width = 25*factor 
    plot_height = 13*factor
    
    timestamp_selected = pd.date_range(start = df_in.index[0] + timedelta(days=1), end = df_in.index[-1], normalize = True, periods = nperiods).tolist()                                                                   # list of dates to display
 
    handles=[]
    df_in.columns = df_in.columns.str.replace('nm','')                               # remove 'nm' from the column names
    for timestamp in timestamp_selected:
        index_date_time=np.where(df_in.index==timestamp.strftime(dateparsingformat))[0] # to check for specific date-time
        idx=index_date_time[:][0]
        plot, = plt.plot(df_in.iloc[idx,1:],                                         # plot
                                    linestyle=':',
                                    marker='o',
                                    markersize=3)                       
        handles.append(plot)

    if xlabel is None:
        xlabel = 'Wavelength [nm]'
    if  ylabel is None:
        ylabel = 'Absorbance' #[abs m$^{-1}$]'        
    if xminlim is None:
        xminlim = 0
    if xmaxlim is None:
        xmaxlim = len(df_in.columns)   
    if yminlim is None:
        ymin = df_in.iloc[:,0].min()
        yminlim = ymin - 0.05*abs(ymin)
    if ymaxlim is None:
        ymax = df_in.iloc[:,0].max()
        ymaxlim = ymax + 0.05*abs(ymax) 
    if xmajlocator is None:
        xmajlocator = round((xmaxlim-xminlim)/8,0) 
    if xminlocator is None:
        xminlocator = xmajlocator/4      
    if ymajlocator is None:
        ymajlocator = round((ymaxlim-yminlim)/4,1) 
    if yminlocator is None:
        yminlocator = ymajlocator/2
        
    plt.style.use ('tableau-colorblind10')                                           # use colors colorblind friendly
    plt.xlabel(xlabel)                                                               # label the x-axis 
    plt.ylabel(ylabel)                                                               # label the y-axis 
    plt.rc('font', family='Arial')                                                   # define the font
    plt.rcParams['axes.labelsize'] = 20                                              # define the label size
    #plt.rcParams['axes.labelweight'] = 'bold'                                       # define the label weight
    plt.gca().tick_params(axis = 'both', which = 'major', length = 5, labelsize = 18)# define the label size of the labels on the axes-ticks (major and minor) and length and color of ticks 
    plt.gca().tick_params(which = 'minor', length = 4, colors = 'k')
    #plt.minorticks_on()                                                              # turn on the minor ticks on both the axes
    plt.gca().xaxis.set_major_locator(MultipleLocator(xmajlocator))                   # define the x-axes major and minor ticks 
    plt.gca().xaxis.set_minor_locator(MultipleLocator(xminlocator))  
    plt.gca().yaxis.set_major_locator(MultipleLocator(ymajlocator))                   # define the y-axes major and minor ticks 
    plt.gca().yaxis.set_minor_locator(MultipleLocator(yminlocator))                     

    #plt.grid(which='major', linestyle=':', linewidth='1', color='gray')              # define the major grid
    #plt.grid(which='minor', linestyle=':', axis = 'x', linewidth='1', color='gray')  # define the minor grid

    plt.legend(handles, timestamp_selected, loc='best', fontsize = 17, markerscale = 2)# define the legend 
                                                                                      
    plt.rcParams['figure.figsize'] = (plot_width, plot_height)                        # define the size of the figure
    plt.tight_layout() 
    fig = plt.gca().get_figure()                                                      # return the figure 
    fig.savefig(output_dir + 'absorbance_spectra_' + str(label_name) + fig_format, dpi = dpi)            # save the figure to the desired format and resolution

    
def abs_basecor(df_in, 
                startwv):
    '''
    function to account for the instrumental baseline drift
    :argument df_in: dataframe in input
    :return: the baseline corrected attenuation dataframe and the standard deviation per each measurement of the chosen range of wavelength   
    '''      
    df_in.columns = [str(col) + 'nm' for col in df_in.columns]
    df_out = df_in.copy()
    header = list(df_out)                            # list of wavelengths; 700-735.5 nm is the wavelength region chosen in this example
    start = df_out.columns.get_loc(startwv)          # get the starting and ending column position of the two wavelengths 
    end = len(header)                                
    med = df_out.iloc[:,start:end].median(axis = 1)    # compute the median of the attenuation values for the columns between start and end    
    std = df_out.iloc[:,start:end].std(axis = 1)
    df_out = df_out.iloc[:,0:start]                   # obtain a subdataset which excludes columns from 700 nm onwards
    df_out = df_out.subtract(med, axis = 0)           # perform the baseline correction    
    return(df_out, med, std)


def makeaplot_nocol(df_in, 
                    output_dir,
                    timestart, 
                    timeend,
                    label_name,
                    title,
                    dateformat = None, 
                    locator = None, 
                    xlabel = None, 
                    ylabel = None, 
                    yminlim = None, 
                    ymaxlim = None, 
                    ymajlocator = None, 
                    yminlocator = None):
    '''
    function to get an even plot structure
    :argument df_in: dataframe in input
    :argument output_dir: directory where storing the results
    :argument timestart, timeend: starting and ending date
    :argument label_name: sample name
    :argument title: figure title
    :argument dateformat: format of the dates to display
    :argument locator: format the date axis automatically, per day, etc. (e.g. "mdates.AutoDateLocator()"/ "mdates.DayLocator()") 
    :argument xlabel, ylabel: x- and y-axes label
    :argument yminlim, ymaxlim: y-axis limits
    :argument yminlocator, ymajlocator: y-axis major and minor ticks 
    :return: the absorbance spectra plot for different dates    
    '''   
    factor = .5
    plot_width = 25*factor                                                           # define the plot dimensions in inches (width and height)
    plot_height = 13*factor
    df_in.plot(style = ['o'],#,'+','s','v','p'],                            # the symbol of the markers  
                        ms = 3,                                                      # the size of the markers
                        label = label_name                                           # the label in the legend
                        ) 
    if dateformat is None:
        dateformat = '%y-%m-%d'
    if locator is None:
        locator = mdates.AutoDateLocator()
    if xlabel is None:
        xlabel = 'Date [yy-mm-dd]'
    if  ylabel is None:
        ylabel = 'Absorbance' #+ '[abs m$^{-1}$]'        
    if yminlim is None:
        ymin = df_in.min()
        yminlim = ymin - 0.05*abs(ymin)
    if ymaxlim is None:
        ymax = df_in.max()
        ymaxlim = ymax + 0.05*abs(ymax) 
    if ymajlocator is None:
        ymajlocator = round((ymaxlim-yminlim)/3,4) 
    if yminlocator is None:
        yminlocator = ymajlocator/3
    
    plt.style.use ('tableau-colorblind10')                                            # use colors colorblind friendly
    plt.xlabel(xlabel)                                                                # label the x-axis 
    plt.ylabel(ylabel)                                                                # label the y-axis 
    plt.rc('font', family='Arial')                                                    # define the font
    plt.rcParams['axes.labelsize'] = 20                                               # define the label size
    #plt.rcParams['axes.labelweight'] = 'bold'                                        # define the label weight
    plt.gca().tick_params(axis = 'both', which = 'major', length = 5, labelsize = 18) # define the label size of the labels on the axes-ticks (major and minor) and length and color of ticks 
    plt.gca().tick_params(which = 'minor', length = 4, colors = 'k')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(dateformat))             #define the format of the date
    plt.gca().xaxis.set_major_locator(locator)                                        # e.g. "AutoDateLocator()"/ "DayLocator()"
    #plt.minorticks_on()                                                              # turn on the minor ticks on both the axes
    plt.gca().set_xlim(timestart,timeend)                                             # define the x-axis range 
    plt.gca().set_ylim(yminlim,ymaxlim)                                               # define the y-axis range
    plt.gca().yaxis.set_major_locator(MultipleLocator(ymajlocator))                   # define the y-axes major and minor ticks 
    plt.gca().yaxis.set_minor_locator(MultipleLocator(yminlocator))                     

    #plt.grid(which='major', linestyle=':', linewidth='1', color='gray')              # define the major grid
    #plt.grid(which='minor', linestyle=':', axis = 'x', linewidth='1', color='gray')  # define the minor grid

    plt.legend(loc='best', fontsize = 17, markerscale = 2)                            # define the legend 
                                                                                      
    plt.rcParams['figure.figsize'] = (plot_width, plot_height)                        # define the size of the figure
    plt.tight_layout() 
    fig = plt.gca().get_figure()                                                      # return the figure 
    fig.savefig(output_dir + title + str(label_name) + fig_format, dpi = dpi)         # save the figure to the desired format and resolution 

def makerollplot(df_in1, df_in2, df_in3,
                 output_dir,
                 col_sel,                
                 timestart, 
                 timeend,
                 label_name,
                 dateformat = None, 
                 locator = None, 
                 xlabel = None, 
                 ylabel = None, 
                 yminlim = None, 
                 ymaxlim = None, 
                 ymajlocator = None, 
                 yminlocator = None):                 
    '''
    function to get a plot of overlapping time series of attenuation data after smoothing them with a moving median filter 
    :argument df_in1, df_in2, df_in3: dataframes in input
    :argument output_dir: directory where storing the results
    :argument col_sel: selected wavelength column
    :argument timestart, timeend: starting and ending date
    :argument label_name: sample name
    :argument dateformat: format of the dates to display
    :argument locator: format the date axis automatically, per day, etc. (e.g. "mdates.AutoDateLocator()"/ "mdates.DayLocator()") 
    :argument xlabel, ylabel: x- and y-axes label
    :argument yminlim, ymaxlim: y-axis limits
    :argument yminlocator, ymajlocator: y-axis major and minor ticks 
    :return: plot of overlapping time series of attenuation data after smoothing them with a moving median filter     
    '''   

    factor = .5
    plot_width = 25*factor # define the plot dimensions in inches (width and height)
    plot_height = 13*factor

    df_in1[col_sel].loc[timestart:timeend].plot(linestyle = '-',                                                
                                                label = '{} {} {} {}'.format(label_name, 'median', str(median_window1_min), 'min'), #add space between words in the label 
                                                color = 'r'
                                                ) 
    df_in2[col_sel].loc[timestart:timeend].plot(linestyle = '-',                            
                                                label = '{} {} {} {}'.format(label_name, 'median', str(median_window2_min), 'min'),  
                                                color = 'c'
                                                ) 

    df_in3[col_sel].loc[timestart:timeend].plot(linestyle = '-',
                                                label = '{} {} {} {}'.format(label_name, 'median', str(median_window3_min), 'min'), 
                                                color = 'magenta' 
                                                )
    if dateformat is None:
        dateformat = '%y-%m-%d'
    if locator is None:
        locator = mdates.AutoDateLocator()
    if xlabel is None:
        xlabel = 'Date [yy-mm-dd]'
    if  ylabel is None:
        ylabel = 'Absorbance' + '$_{' + col_sel + '}$' #+ '[abs m$^{-1}$]'        
    if yminlim is None:
        ymin = df_in1[col_sel].min()
        yminlim = ymin - 0.05*abs(ymin)
    if ymaxlim is None:
        ymax = df_in1[col_sel].max()
        ymaxlim = ymax + 0.05*abs(ymax) 
    if ymajlocator is None:
        ymajlocator = round((ymaxlim-yminlim)/6,2) 
    if yminlocator is None:
        yminlocator = ymajlocator/3
    plt.style.use ('tableau-colorblind10')                                           # use colors colorblind friendly
    plt.xlabel(xlabel)                                                               # label the x-axis 
    plt.ylabel(ylabel)                                                               # label the y-axis 
    plt.rc('font', family='Arial')                                                   # define the font
    plt.rcParams['axes.labelsize'] = 20                                              # define the label size
    #plt.rcParams['axes.labelweight'] = 'bold'                                       # define the label weight
    plt.gca().tick_params(axis = 'both', which = 'major', length = 5, labelsize = 18)# define the label size of the labels on the axes-ticks (major and minor) and length and color of ticks 
    plt.gca().tick_params(which = 'minor', length = 4, colors = 'k')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(dateformat))             #define the format of the date
    plt.gca().xaxis.set_major_locator(locator)                                        # e.g. "AutoDateLocator"/ "DayLocator"
    #plt.minorticks_on()                                                              # turn on the minor ticks on both the axes
    plt.gca().set_xlim(timestart,timeend)                                             # define the x-axis range 
    plt.gca().set_ylim(yminlim,ymaxlim)                                               # define the y-axis range
    plt.gca().yaxis.set_major_locator(MultipleLocator(ymajlocator))                   # define the y-axes major and minor ticks 
    plt.gca().yaxis.set_minor_locator(MultipleLocator(yminlocator))                     

    #plt.grid(which='major', linestyle=':', linewidth='1', color='gray')              # define the major grid
    #plt.grid(which='minor', linestyle=':', axis = 'x', linewidth='1', color='gray')  # define the minor grid

    plt.legend(loc='best', fontsize = 17, markerscale = 2)                            # define the legend 
                                                                                      
    plt.rcParams['figure.figsize'] = (plot_width, plot_height)                        # define the size of the figure
    plt.tight_layout() 
    fig = plt.gca().get_figure()                                                      # return the figure 
    fig.savefig(output_dir + 'absorbance_data_median_filtered_' + str(label_name) + fig_format, dpi = dpi)   # save the figure to the desired format and resolution
    
def makeakdeplot(df_in, 
                 output_dir,
                 label_name,
                 col_sel = None, 
                 xlabel = None, 
                 ylabel = None, 
                 yminlim = None, 
                 ymaxlim = None, 
                 ymajlocator = None, 
                 yminlocator = None):                  
    '''
    function to get a kde plot
    :argument df_in: dataframe in input
    :argument output_dir: directory where storing the results
    :argument label_name: sample name
    :argument col_sel: selected wavelength columns
    :argument xlabel, ylabel: x- and y-axes label
    :argument yminlim, ymaxlim: y-axis limits
    :argument yminlocator, ymajlocator: y-axis major and minor ticks 
    :return: gaussian kde plot at specific wavelengths
    '''
    factor = .5
    plot_width = 25*factor # define the plot dimensions in inches (width and height)
    plot_height = 13*factor
        
    if col_sel is None:
        col_sel = ['255 nm','275 nm','295 nm','350 nm','400 nm','697.5 nm']
    for i in col_sel:
        sns.kdeplot(df_in[i], kernel='gau') 
       
    if xlabel is None:
        xlabel = 'Absorbance' #[abs m$^{-1}$]'
    if  ylabel is None:
        ylabel = 'Gaussian KDE [dimesionless]'        
    if yminlim is None:
        ymin = (df_in[col_sel].min()).min()                                         
        yminlim = ymin - 0.05*abs(ymin)
    if ymaxlim is None:
        ymax = (df_in[col_sel].max()).max()
        ymaxlim = ymax + 0.05*abs(ymax) 
    if ymajlocator is None:
        ymajlocator = round((ymaxlim-yminlim)/6,2) 
    if yminlocator is None:
        yminlocator = ymajlocator/3

    plt.style.use ('tableau-colorblind10')                                           # use colors colorblind friendly
    plt.xlabel(xlabel)                                                               # label the x-axis 
    plt.ylabel(ylabel)                                                               # label the y-axis 
    plt.rc('font', family='Arial')                                                   # define the font
    plt.rcParams['axes.labelsize'] = 20                                              # define the label size
    #plt.rcParams['axes.labelweight'] = 'bold'                                       # define the label weight
    plt.gca().tick_params(axis = 'both', which = 'major', length = 5, labelsize = 18)# define the label size of the labels on the axes-ticks (major and minor) and length and color of ticks 
    plt.gca().tick_params(which = 'minor', length = 4, colors = 'k')
    #plt.minorticks_on()                                                              # turn on the minor ticks on both the axes
    plt.gca().set_ylim(yminlim,ymaxlim)                                               # define the y-axis range
    plt.gca().yaxis.set_major_locator(MultipleLocator(ymajlocator))                   # define the y-axes major and minor ticks 
    plt.gca().yaxis.set_minor_locator(MultipleLocator(yminlocator))                     

    #plt.grid(which='major', linestyle=':', linewidth='1', color='gray')              # define the major grid
    #plt.grid(which='minor', linestyle=':', axis = 'x', linewidth='1', color='gray')  # define the minor grid

    plt.legend(loc='best', fontsize = 17, markerscale = 2)                            # define the legend 
                                                                                      
    plt.rcParams['figure.figsize'] = (plot_width, plot_height)                        # define the size of the figure
    #plt.tight_layout() 
    fig = plt.gca().get_figure()                                                      # return the figure 
    fig.savefig(output_dir + 'kde_' + str(label_name) + fig_format, dpi = dpi)        # save the figure to the desired format and resolution

    
def abs_slope_ratio(df_in,
                    sampling_frequency):
    '''
    function to get a dataframe with the slope values at shorter wavelengths (S275-295), at longer wavelengths (S350-400) and their ratio  (slope ratio)
    :argument df_in: dataframe in input
    :argument sampling_frequency: sampling frequency when computing the slope ratio
    :return: dataframe with S275-295, S350-400 and slope ratio (SR) at the sampling frequency specified
    '''
    df_in = df_in.reset_index()                                  # restore the original index as row
    df_in = df_in.T                                              # transpose the data frame  
    df_in = df_in.reset_index()                                  # restore the original index as row
    headers = df_in.iloc[0]                                      # rename the dates as headers                                     
    df_in = pd.DataFrame(df_in.values[1:], columns=headers)      # convert the first row to header
    df_in = df_in.rename(columns={"Timestamp": "wl"})            # rename the first column as "wl" = wavelength vector
    df_in['wl'] = df_in['wl'].replace({'nm':''},regex=True)      # remove "nm" from the wavelength vector
    wl = df_in['wl'].apply(pd.to_numeric)                        # convert the wavelength vector to numeric  

    iteration = len(df_in.columns)                               # number of loop iterations
    print('number of iterations',iteration)
    empty_matrix = np.zeros((iteration, 3))                      # create an empty matrix to fill in with the parameters s275_295, s350_400,SR for each datetime

    wl_275_295 = np.linspace(275,295,25)                         # create an array of 25 evenly spaced numbers over the specified interval
    wl_350_400 = np.linspace(350,400,25) 
    wl_275_295_resh = np.array(wl_275_295).reshape((-1, 1))      # reshape as column
    wl_350_400_resh = np.array(wl_350_400).reshape((-1, 1))

    counter = 0                                                  # use a counter in the loop to keep track of the iteration number
    for i in range(0,iteration,sampling_frequency): 
        counter = i                                           
        print(counter)                                 
        absorbance = df_in.iloc[:,i]                             # subset the absorbance values   
        sf = interpolate.interp1d(wl, absorbance,kind='cubic')# spline interpolation of third order to get as many absorbance values as the array of 25 wavelengths
        absorbance_275_295 = sf(wl_275_295)
        absorbance_350_400 = sf(wl_350_400)
        absorbance_275_295_log_resh = np.log(absorbance_275_295).reshape((-1, 1))
        absorbance_350_400_log_resh = np.log(absorbance_350_400).reshape((-1, 1))
        lm_275_295 = LinearRegression().fit(wl_275_295_resh, absorbance_275_295_log_resh) # perform a linear regression and calculate the slopes in the two wavelength ranges
        slope_275_295 = lm_275_295.coef_[0]
        lm_350_400 = LinearRegression().fit(wl_350_400_resh, absorbance_350_400_log_resh)
        slope_350_400 = lm_350_400.coef_[0]
        print('slope 275-295:',slope_275_295)
        print('slope 350-400:',slope_350_400)
        sr = slope_275_295/slope_350_400                         # compute the slope ratio 
        print('SR:',sr) 
        empty_matrix[i, 0] = slope_275_295                       # fill in the empty matrix with the calculated parameters
        empty_matrix[i, 1] = slope_350_400  
        empty_matrix[i, 2] = sr    

    sr_data = pd.DataFrame(empty_matrix, index=headers.iloc[0:], columns=['s275_295','s350_400','SR']) # create a dataframe with the calculated parameters
    #print(sr_data)
    df_out = sr_data[sr_data['SR'] != 0][1:]                     # keep only the rows in which the slope ratio is different from zero 
    df_out.index = pd.to_datetime(df_out.index)
    print(df_out)    
    return(df_out)


def outlier_id_drop_iqr(df_in, 
                        output_dir1,
                        output_dir2,
                        splitstrings,
                        timestart, 
                        timeend,
                        dateparsingformat,
                        label_name):
    '''
    function to split the slope ratio dataframe in different periods and identify the outliers on the basis of the interquartile range in  these periods 
    :argument df_in: dataframe in input
    :argument output_dir1: directory where storing the dataframe without outliers
    :argument output_dir2: directory where storing the outliers
    :argument splitstrings: dates to use to split the dataset in periods
    :argument timestart, timeend: starting and ending date
    :argument dateparsingformat: format of the dates 
    :argument label_name: sample name    
    :return: the slope ratio dataframe in different periods (dflist), lower and upper limits of the interquartile range used
    to detect the outliers (out1) and the outlier percentage (out2)    
    '''
    ### SPLIT THE DATAFRAME INTO DIFFERENT PERIODS
    splitstrs2 = splitstrings.copy()
    splitstrs2.insert(0,timestart)                                    # add start and end dates to the dates we want to use to split the periods
    splitstrs2.append(timeend)
    split_date = [dt.strptime(splitdates,dateparsingformat) for splitdates in splitstrs2] # convert to datetime format
    #print('dates used to split the dataframe in periods: ', splitstrs)
    nperiods = len(splitstrs2) - 1   
    dflist = []                                                       # create an empty list 
    for i in range(nperiods):                                         # append the dataframes obtained for each period to the empty list  
        df_periods = df_in.loc[(df_in.index >= split_date[i]) & (df_in.index < split_date[i+1])]
        #print('period '+ str(i+1) + ': ' + dt.strftime(split_date[i], dateparsingformat) + ' - ' + dt.strftime(split_date[i+1], dateparsingformat))
        dflist.append(df_periods)

    ### CALCULATE LOWER AND UPPER LIMITS OF THE INTERQUARTILE RANGE

    out1 = pd.DataFrame(columns=['low_lim','up_lim'])                 # create an empty table with two columns and the same number of rows as the periods and fill it with NA
    out1['low_lim'] = np.repeat('NA', nperiods, axis = 0)             # rename the two columns
    out1['up_lim'] =  np.repeat('NA', nperiods, axis = 0)

    for i in range(nperiods):                                         # for each period: 
        df = dflist[i]
        q1 = np.percentile(df['SR'], 25, interpolation = 'midpoint')  # calculate the quartiles Q1, Q2, Q3
        q2 = np.percentile(df['SR'], 50, interpolation = 'midpoint')
        q3 = np.percentile(df['SR'], 75, interpolation = 'midpoint')
        iqr = q3-q1                                                   # calculate the interquartile range IQR
        low_lim = q1 - 1.5 * iqr                                      # find the lower and upper limits 
        up_lim = q3 + 1.5 * iqr
        out1['low_lim'] [i] = low_lim                                 # get all the lower and upper limits in the empty table
        out1['up_lim'] [i] = up_lim 

    ### CALCULATE OUTLIER PERCENTAGES

    out2 = pd.DataFrame(columns=['number_outliers','outliers (%)'])   # create an empty table with two columns and the same number of rows as the periods and fill it with NA                
    out2['number_outliers'] = np.repeat('NA', nperiods, axis=0)       # rename the two columns
    out2['outliers (%)'] = np.repeat('NA', nperiods, axis=0)

    dflist_noout = []                   # create an empty list 
    for i in range(nperiods):                                         # for each period:  
        df = dflist[i]   
        df_noout = df[(df_in.SR > out1['low_lim'][i]) & (df.SR < out1['up_lim'][i])] # remove SR values lower than the lower limit and greater than the upper limit (= outliers)
        ntot = len(df)                                                # number of measurements
        df_out = df[(df.SR < out1['low_lim'][i]) | (df.SR > out1['up_lim'][i])] # find out which SR values are lower than the lower limit and greater than the upper limit (= outliers)
        nout = len(df_out)                                            # number of outliers
        out2['number_outliers'] [i] = nout                            # get the number of outliers in a table together with its relative percentage                             
        out2['outliers (%)'] [i] = nout/ntot * 100 
        df_noout.to_csv(output_dir1 + 'df_sr_' + str(label_name) + '_' + '1_no_outliers' + str(i) + '.csv', sep=sep, decimal=decimal, index=True)                                                           # save the not outliers for the different datasets   
        df_out.to_csv(output_dir2 + 'df_sr_' + str(label_name) + '_' + '1_outliers' + str(i) + '.csv', sep=sep, decimal=decimal, index=True)                                                           # save the outliers for the different datasets  
        dflist_noout.append(df_noout)

    return(dflist, out1, out2) 


def makeaoutplot(df_in1,
                 df_in2,
                 output_dir,
                 col_sel,                
                 timestart, 
                 timeend,
                 dateparsingformat,
                 splitstrings,
                 label_name,
                 dateformat = None, 
                 locator = None, 
                 xlabel = None, 
                 ylabel = None, 
                 plabel = None, 
                 yminlim = None, 
                 ymaxlim = None, 
                 ymajlocator = None, 
                 yminlocator = None,
                 ymajlocator_box = None,                      
                 yminlocator_box = None):                 
    '''
    function to get a subplot of slope ratio values and a subplot of boxplots computed on the basis of the slope ratio values (interquartile range) 
    :argument df_in: dataframe in input (dflist from the function "outlier_id_drop_iqr")
    :argument df_in2: dataframe in input ("output_no_outliers" folder)
    :argument output_dir: directory where storing the results
    :argument col_sel: selected column
    :argument timestart, timeend: starting and ending date
    :argument dateparsingformat: format of the dates 
    :argument splitstrings: dates to use to split the dataset in periods
    :argument label_name: sample name
    :argument dateformat: format of the dates to display
    :argument locator: format the date axis automatically, per day, etc. (e.g. "mdates.AutoDateLocator()"/ "mdates.DayLocator()") 
    :argument xlabel, ylabel: x- and y-axes label
    :argument plabel: y coordinate of the period labels
    :argument yminlim, ymaxlim: y-axis limits
    :argument yminlocator, ymajlocator: y-axis major and minor ticks 
    :argument yminlocator_box, ymajlocator_box: y-axis ticks (major and minor) for boxplots
    :return: two subplots with slope ratio values and its boxplots    
    '''  
    ### PLOT BEFORE REMOVING OUTLIERS:

    factor = .5
    plot_width = 25*factor # define the plot dimensions in inches (width and height)
    plot_height = 13*factor

    fig, axs = plt.subplots(2, gridspec_kw={'hspace': 0})                 # subplots sharing x-axis; add "sharey=True" if they 
                                                                          # share also the y-axis; "hspace" to reduce the
                                                                          # space between vertical subplots

    splitstrs2 = splitstrings.copy()
    splitstrs2.insert(0,timestart)                                        # add start and end dates to the dates we want to use to split the periods
    splitstrs2.append(timeend)
    split_date = [dt.strptime(splitdates,dateparsingformat) for splitdates in splitstrs2] # convert to datetime format
    #print('dates used to split the dataframe in periods: ', splitstrs2)
    nperiods = len(splitstrs2) - 1   

    tst = dt.strptime(timestart,dateparsingformat)                         # to set the position of the boxplots on the x-axis,
    tend = dt.strptime(timeend,dateparsingformat)                          # create a datetime object from the string of the                          
    
    bplist = []                                                            # create an empty list 
    for i in range(nperiods):                                              # append the dataframes obtained for each period to the empty list  
        bp = df_in1[i].index.mean().strftime(dateparsingformat)
        #print('period '+ str(i+1) + ': ' + dt.strftime(split_date[i], dateparsingformat) + ' - ' + dt.strftime(split_date[i+1], dateparsingformat))
        bplist.append(bp)
    #print(bplist)                                                         # the initial and final date in the plot and do the
    tpos=[dt.strptime(dstr,dateparsingformat) for dstr in bplist]          # same for the dates used for the boxplots positions; 
    tdelta = tend - tst                                                    # use the time difference between the initial and 
    pos = [ (tstep - tst)/tdelta for tstep in tpos ]                       # final date to get the boxplots positions as fractions
                                                                           # of the total time span
    for i in range(nperiods):                                              # append the dataframes obtained for each period to the empty list  
        df = df_in1[i]
        axs[0].boxplot (df[col_sel],                                       # plot boxplots for the different periods
                 showfliers=True,                                          # show or less outliers
                 widths=0.03,                                              # adjust the width of the box
                 whis=1.5,                                                 # set the whisker length 
                 patch_artist=True, boxprops={'facecolor': 'c','edgecolor': 'k'}, # fill the boxplot and color the border
                 medianprops = dict(linestyle='-', linewidth=1, color='k'),# color and size the median line
                 flierprops = dict(marker='*', markerfacecolor='c', markersize=2,markeredgecolor='c'), # marker symbol,color,size
                 positions = [pos[i]]                                      # move the boxplots along the x-axis
                 ) 
        df[col_sel].loc[timestart:timeend].plot(style = ['*'],             # the symbol of the markers  
                                                ms = 3,                    # the size of the markers
                                                label = label_name,        # the label in the legend
                                                ax=axs[1])
    axs[0].set_xlim(0,1)
    axs[0].set_xticklabels([])

    
    if dateformat is None:
        dateformat = '%y-%m-%d'
    if locator is None:
        locator = mdates.AutoDateLocator()
    if xlabel is None:
        xlabel = 'Date [yy-mm-dd]'
    if  ylabel is None:
        ylabel = 'Slope ratio [dimensionless]'        
    if yminlim is None:
        ymin = df[col_sel].min()
        yminlim = ymin - 0.05*abs(ymin)
    if ymaxlim is None:
        ymax = df[col_sel].max()
        ymaxlim = ymax + 0.05*abs(ymax) 
    if ymajlocator is None:
        ymajlocator = round((ymaxlim-yminlim)/6,2) 
    if yminlocator is None:
        yminlocator = ymajlocator/3
    if plabel is None:
        plabel = yminlim + 0.01*yminlim                                                               
    if ymajlocator_box is None:
        ymajlocator_box = ymajlocator                                                       
    if yminlocator_box is None:
        yminlocator_box = yminlocator                                                   

    from cycler import cycler
    plt.rcParams['axes.prop_cycle'] = cycler(color='c')

    plt.xlabel(xlabel)                                                      # label the x-axis 
    plt.ylabel(ylabel)                                                      # label the y-axis 
    plt.gca().yaxis.set_label_coords(-0.06,1)                               # change ylabel position
    plt.rc('font', family='Arial')                                          # define the font
    plt.rcParams['axes.labelsize'] = 20                                     # define the label size
    #plt.rcParams['axes.labelweight'] = 'bold'                              # define the label weight
    plt.gca().tick_params(axis = 'both', which = 'major', length = 5, labelsize = 18)# define the label size of the labels on the axes-ticks (major and minor) and length and color of ticks 
    plt.gca().tick_params(which = 'minor', length = 4, colors = 'k')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(dateformat))   # define the format of the date
    plt.gca().xaxis.set_major_locator(locator)                              # e.g. "AutoDateLocator"/ "DayLocator"
    #plt.minorticks_on()                                                    # turn on the minor ticks on both the axes
    plt.gca().set_xlim(timestart,timeend)                                   # define the x-axis range 
    plt.gca().set_ylim(yminlim,ymaxlim)                                     # define the y-axis range
    plt.gca().yaxis.set_major_locator(MultipleLocator(ymajlocator))         # define the y-axes major and minor ticks 
    plt.gca().yaxis.set_minor_locator(MultipleLocator(yminlocator))                     

    #plt.grid(which='major', linestyle=':', linewidth='1', color='gray')    # define the major grid
    #plt.grid(which='minor', linestyle=':', axis = 'x', linewidth='1', color='gray')  # define the minor grid

    #plt.legend(loc='best', fontsize = 17, markerscale = 2)                 # define the legend 

    d = {}                                                                  # create a list of period labels
    for x in range(1, nperiods + 1):
        d['P%d' % x] = x   

    labelplist = []    
    for key, value in sorted(d.items()):
        labelplist.append(key)
    #print(labelplist)

    for i in range(nperiods):     
        plt.text(bplist[i], plabel, labelplist[i], fontsize=16, horizontalalignment='right', color ='black', fontweight = 'bold') # add text to distinguish different periods

    for i in range(nperiods): 
         plt.axvline(x = splitstrs2[i], color='black', linestyle=':')       # add vertical lines to split the plot in 

    axs[0].yaxis.set_major_locator(MultipleLocator(ymajlocator_box))        # define the y-axes major and minor ticks 
    axs[0].yaxis.set_minor_locator(MultipleLocator(yminlocator_box))        # as multiples of a number
    axs[0].tick_params(axis = 'both', which = 'major', length = 5, labelsize = 18) # define the label size of the labels on the axes-ticks 
    axs[0].tick_params(which = 'minor', length = 4, colors = 'k')

    plt.rcParams['figure.figsize'] = (plot_width, plot_height)              # define the size of the figure
    plt.tight_layout() 
    fig = plt.gca().get_figure()                                            # return the figure     

    fig.suptitle('Before outlier removal', y=1)              # add a title
    fig.savefig(output_dir + 'sr_boxplots_sw_periods_before_outliers_removal' + fig_format, dpi = dpi) # save the figure to the desired format and resolution
    
    
    ### PLOT AFTER REMOVING OUTLIERS:
    fig, axs = plt.subplots(2, gridspec_kw={'hspace': 0})                   # subplots sharing x-axis; add "sharey=True" if they 
                                                                            # share also the y-axis; "hspace" to reduce the
                                                                            # space between vertical subplots
    factor=.5
    plot_width=25*factor # inches; define the plot dimensions (width and height)
    plot_height=13*factor # inches

    tst = dt.strptime(timestart,dateparsingformat)                          # to set the position of the boxplots on the x-axis,
    tend = dt.strptime(timeend,dateparsingformat)                           # create a datetime object from the string of the                          
                                                                            # the initial and final date in the plot and do the  
    tpos=[dt.strptime(dstr,dateparsingformat) for dstr in bplist]           # same for the dates used for the boxplots positions; 
    tdelta = tend - tst                                                     # use the time difference between the initial and 
    pos = [ (tstep - tst)/tdelta for tstep in tpos ]                        # final date to get the boxplots positions as fractions
                                                                            # of the total time span

    for icnt, filename in enumerate(glob.glob(df_in2 + '*.csv')):
        df=pd.read_csv(filename, sep=sep,header=header,index_col=0)    
        axs[0].boxplot (df[col_sel],                                        # plot boxplots for the different periods
                 showfliers=True,                                           # show or less outliers
                 widths=0.03,                                               # adjust the width of the box
                 whis=1.5,                                                  # set the whisker length 
                 patch_artist=True, boxprops={'facecolor': 'c','edgecolor': 'k'}, # fill the boxplot and color the border
                 medianprops = dict(linestyle='-', linewidth=1, color='k'),       # color and size the median line
                 flierprops = dict(marker='*', markerfacecolor='c', markersize=2,markeredgecolor='c'), # marker symbol,color,size
                 positions = [pos[icnt]]                                    # move the boxplots along the x-axis
                 )
    axs[0].set_xlim(0,1)
    axs[0].set_xticklabels([])

    for icnt, filename in enumerate(glob.glob(df_in2 + '*.csv')):    
        df=pd.read_csv(filename, sep=sep,header=header,index_col=0) 
        df.index = pd.to_datetime(df.index, format= dateparsingformat)
        df[col_sel].loc[timestart:timeend].plot(style = ['*'],               # the symbol of the markers  
                                                ms = 3,                      # the size of the markers 
                                                label = label_name,          # the label in the legend
                                                ax=axs[1])

    from cycler import cycler
    plt.rcParams['axes.prop_cycle'] = cycler(color='c')

    plt.xlabel(xlabel)                                                       # label the x-axis 
    plt.ylabel(ylabel)                                                       # label the y-axis 
    plt.gca().yaxis.set_label_coords(-0.06,1)                                # change ylabel position
    plt.rc('font', family='Arial')                                           # define the font
    plt.rcParams['axes.labelsize'] = 20                                      # define the label size
    #plt.rcParams['axes.labelweight'] = 'bold'                               # define the label weight
    plt.gca().tick_params(axis = 'both', which = 'major', length = 5, labelsize = 18) # define the label size of the labels on the axes-ticks (major and minor) and length and color of ticks 
    plt.gca().tick_params(which = 'minor', length = 4, colors = 'k')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(dateformat))    #define the format of the date
    plt.gca().xaxis.set_major_locator(locator)                               # e.g. "AutoDateLocator"/ "DayLocator"
    #plt.minorticks_on()                                                     # turn on the minor ticks on both the axes
    plt.gca().set_xlim(timestart,timeend)                                    # define the x-axis range 
    plt.gca().set_ylim(yminlim,ymaxlim)                                      # define the y-axis range
    plt.gca().yaxis.set_major_locator(MultipleLocator(ymajlocator))          # define the y-axes major and minor ticks 
    plt.gca().yaxis.set_minor_locator(MultipleLocator(yminlocator))                     

    #plt.grid(which='major', linestyle=':', linewidth='1', color='gray')     # define the major grid
    #plt.grid(which='minor', linestyle=':', axis = 'x', linewidth='1', color='gray') # define the minor grid

    #plt.legend(loc='best', fontsize = 17, markerscale = 2)                  # define the legend 

    for i in range(nperiods):     
        plt.text(bplist[i], plabel, labelplist[i], fontsize=16, horizontalalignment='right', color ='black', fontweight = 'bold') # add text to distinguish different periods

    for i in range(nperiods): 
         plt.axvline(x = splitstrs2[i], color='black', linestyle=':')        # add vertical lines to split the plot in 

    axs[0].yaxis.set_major_locator(MultipleLocator(ymajlocator_box))         # define the y-axes major and minor ticks 
    axs[0].yaxis.set_minor_locator(MultipleLocator(yminlocator_box))         # as multiples of a number
    axs[0].tick_params(axis = 'both', which = 'major', length = 5, labelsize = 18) # define the label size of the labels on the axes-ticks 
    axs[0].tick_params(which = 'minor', length = 4, colors = 'k')

    plt.rcParams['figure.figsize'] = (plot_width, plot_height)               # define the size of the figure
    plt.tight_layout() 
    fig = plt.gca().get_figure()                                             # return the figure     

    fig.suptitle('After outlier removal', y=1)                # add a title
    fig.savefig(output_dir + 'sr_boxplots_sw_periods_after_outliers_removal' + fig_format, dpi = dpi) # save the figure to the desired format and resolution      
    

def outliers_id_drop(df_in1, 
                     df_in2, 
                     output_dir,
                     col_sel):
    '''
    function to label outliers and events based on user knowledge and remove the known outliers
    :argument df_in1: dataframe in input (df_bc from the function "abs_basecor")
    :argument df_in2: dataframe in input (df with events)
    :argument output_dir: directory where storing the results
    :argument col_sel: selected wavelength column
    return: the dataframe of events (df_out1) and the baseline corrected dataframe after removing the specified events (df_out2)
    '''
    ### VISUALISE SPECIFIC EVENT TYPOLOGIES
    df_ev = pd.read_csv(df_in2, sep = sep2, header = header, parse_dates=[['start_date', 'start_time'], ['end_date', 'end_time']])
    df_ev = df_ev.dropna(subset=['event_code'])                              # remove all the rows that do not have an event code

    x_ev = df_ev['start_date_start_time'] + ((df_ev['end_date_end_time']-df_ev['start_date_start_time'])/2) # compute the average datetime of the event
    #print('axis-x:', x_ev)

    y_ev = []                                                                 # compute at which y-axis coordinate insert the symbol
    for i in range(0, len(x_ev)):
        timestart2 = df_ev['start_date_start_time'] [i]
        timeend2 = df_ev['end_date_end_time'] [i]    
        ym = median(df_in1[col_sel].loc[timestart2:timeend2])                 # calculate the median per each time period correspondant to an event
        ystart = df_in1[col_sel].iloc[df_in1.index.get_loc(timestart2, method='nearest')] # get the absorbance value at the closest timestamp to the event starting date
        if ym > ystart:                                                       # add one unit or subtract one unit according to the fact that the median absorbance value during the event is a positive value greater or lower than one for visualisation purposes
            ysel = ym + 1
        else:
            ysel = ym - 1  
        y_ev.append(ysel)
    #print('axis-y:', y_ev)

    df_ev['middle_date_time'], df_ev ['median_abs_middle_date_time'] = [x_ev, y_ev] # add two columns to the events dataframe, including the average date of the event and the median absorbance value for that event plus/minus 1 unit
    ev = df_ev.loc[df_ev.loc[df_ev['event_code'] == evdrop].index]
    df_ev = df_ev[['middle_date_time', 'median_abs_middle_date_time', 'event_code']] # subset the dataframe 
    df_ev.set_index('middle_date_time', inplace=True)                          # set the average date as index
    
    ### DROP SPECIFIC EVENT TYPOLOGIES

    ev.reset_index(inplace = True)

    df_drop = df_in1.copy()
    for i in range(0, len(ev)):
        mask = (df_drop.index >= ev['start_date_start_time'][i]) & (df_drop.index <= ev['end_date_end_time'][i])
        df_drop = df_drop.drop(df_drop[mask].index)
    df_drop.to_csv(output_dir + 'df_noselectedevents.csv', index = True)
    df_out1 = df_ev
    df_out2 = df_drop
    return(df_out1, df_out2)   


def makeaplotev(df_in1,
                df_in2,
                output_dir,
                col_sel,                
                timestart, 
                timeend,
                dateparsingformat,
                label_name,
                title,
                dateformat = None, 
                locator = None, 
                xlabel = None, 
                ylabel = None, 
                plabel = None, 
                yminlim = None, 
                ymaxlim = None, 
                ymajlocator = None, 
                yminlocator = None,
                ymajlocator_box = None,                      
                yminlocator_box = None): 
       
    '''
    function to get a plot of the attenuation data time series and events displayed
    :argument df_in1: dataframe in input (df_out1 from the function "outliers_id_drop")
    :argument df_in2: dataframe in input (df_out2 from the function "outliers_id_drop")
    :argument output_dir: directory where storing the results
    :argument col_sel: selected column
    :argument timestart, timeend: starting and ending date
    :argument dateparsingformat: format of the dates 
    :argument label_name: sample name
    :argument title: figure title
    :argument dateformat: format of the dates to display
    :argument locator: format the date axis automatically, per day, etc. (e.g. "mdates.AutoDateLocator()"/ "mdates.DayLocator()") 
    :argument xlabel, ylabel: x- and y-axes label
    :argument plabel: y coordinate of the period labels
    :argument yminlim, ymaxlim: y-axis limits
    :argument yminlocator, ymajlocator: y-axis major and minor ticks 
    :argument yminlocator_box, ymajlocator_box: y-axis ticks (major and minor) for boxplots
    :return: plot of attenuation data with events or without specific events
    '''  
       
    factor = .5
    plot_width = 25*factor                                                           # define the plot dimensions in inches (width and height)
    plot_height = 13*factor
    df_in1[col_sel].loc[timestart:timeend].plot(style = ['o'],#,'+','s','v','p'],    # the symbol of the markers  
                                                ms = 3,                              # the size of the markers
                                                label= label_name,                   # the label in the legend
                                                alpha = 0.8)                         # transparency 
    for i in range(0, nevents):
          df_in2.loc[df_in2['event_code'] == i+1]['median_abs_middle_date_time'].plot(style = symbols[i], 
                                                                                      ms = 7,
                                                                                      label = '{} {}'.format(sample_name, 'event' + str(i+1)), # add space between words in the label 
                                                                                      color = 'blue') 
    if dateformat is None:
        dateformat = '%y-%m-%d'
    if locator is None:
        locator = mdates.AutoDateLocator()
    if xlabel is None:
        xlabel = 'Date [yy-mm-dd]'
    if  ylabel is None:
        ylabel = 'Absorbance' + '$_{' + col_sel + '}$' #+ '[abs m$^{-1}$]'       
    if yminlim is None:
        ymin = df_in1[col_sel].min()
        yminlim = ymin - 0.05*abs(ymin)
    if ymaxlim is None:
        ymax = df_in1[col_sel].max()
        ymaxlim = ymax + 0.05*abs(ymax) 
    if ymajlocator is None:
        ymajlocator = round((ymaxlim-yminlim)/6,2) 
    if yminlocator is None:
        yminlocator = ymajlocator/3
    if plabel is None:
        plabel = yminlim + 0.01*yminlim                                                               
    if ymajlocator_box is None:
        ymajlocator_box = ymajlocator                                                       
    if yminlocator_box is None:
        yminlocator_box = yminlocator  
            
    plt.style.use ('tableau-colorblind10')                                            # use colors colorblind friendly
    plt.xlabel(xlabel)                                                                # label the x-axis 
    plt.ylabel(ylabel)                                                                # label the y-axis 
    plt.rc('font', family='Arial')                                                    # define the font
    plt.rcParams['axes.labelsize'] = 20                                               # define the label size
    #plt.rcParams['axes.labelweight'] = 'bold'                                        # define the label weight
    plt.gca().tick_params(axis = 'both', which = 'major', length = 5, labelsize = 18) # define the label size of the labels on the axes-ticks (major and minor) and length and color of ticks 
    plt.gca().tick_params(which = 'minor', length = 4, colors = 'k')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(dateformat))             # define the format of the date
    plt.gca().xaxis.set_major_locator(locator)                                        # e.g. "AutoDateLocator"/ "DayLocator"
    #plt.minorticks_on()                                                              # turn on the minor ticks on both the axes
    plt.gca().set_xlim(timestart,timeend)                                             # define the x-axis range 
    plt.gca().set_ylim(yminlim,ymaxlim)                                               # define the y-axis range
    plt.gca().yaxis.set_major_locator(MultipleLocator(ymajlocator))                   # define the y-axes major and minor ticks 
    plt.gca().yaxis.set_minor_locator(MultipleLocator(yminlocator))                     

    #plt.grid(which='major', linestyle=':', linewidth='1', color='gray')              # define the major grid
    #plt.grid(which='minor', linestyle=':', axis = 'x', linewidth='1', color='gray')  # define the minor grid

    plt.legend(loc='best', fontsize = 17, markerscale = 1.2)                          # define the legend 

    plt.rcParams['figure.figsize'] = (plot_width, plot_height)                        # define the size of the figure
    plt.tight_layout() 
    fig = plt.gca().get_figure()                                                      # return the figure 
    fig.savefig(output_dir + title + fig_format, dpi = dpi)                           # save the figure to the desired format and resolution
    
    
def abs_ratio(df_in, 
              date_ref_start,
              date_ref_end,
              dateparsingformat,
              wavelength1,
              wavelength2,              
              date_string):
    '''
    function to calculate the ratio of absorbance data at two different wavelengths and compute the percentage change in relation to a reference period 
    :argument df_in: dataframe in input 
    :argument date_ref_start: starting date of the reference period
    :argument date_ref_end: ending date of the reference period
    :argument dateparsingformat: format of the dates
    :argument wavelength1: wavelength numerator
    :argument wavelength2: wavelength denominator
    :argument date_string: date where to look at the percentage change compared to the reference period
    :return: dataframe with the ratio of absorbance data at two different wavelengths (df_out1) and with percentage changes of the ratio of absorbance data in relation to a reference period (df_out2)
    '''    
    df_ratio = df_in[wavelength1]/df_in[wavelength2]
    
    ### COMPUTE THE PERCENTAGE CHANGES OF THE ABSORBANCE RATIOS DATA IN RELATION TO A REFERENCE PERIOD AND PLOT IT
    # Define a reference period:
    date_ref_start1 = str(df_in.iloc[df_in.index.get_loc(date_ref_start, method='nearest')].name) # find the closest date to the date_ref_start and the second closest date to the date_ref_end available in the dataframe
    date_ref_end1 = str(df_in.iloc[df_in.index.get_loc(date_ref_end, method='nearest')+1].name)
    timestamp_selected = pd.date_range(start = date_ref_start1, end = date_ref_end1, periods = 2).tolist()    
    idx_vector = []
    for timestamp in timestamp_selected:
            index_date_time=np.where(df_in.index == timestamp.strftime(dateparsingformat))[0]     # to check for specific date-time
            idx=index_date_time[:][0]
            idx_vector.append(idx)
    #print('idx_vector:',idx_vector)

    df_ref = df_ratio.copy()
    df_ref = df_ref[idx_vector[0]:idx_vector[1]]

    # Normalize the data by the average of the reference period and multiply by 100 to get %change:
    df_ref_av = df_ref.mean()                                                                     # compute the average of the vector
    df_sub_ref_av = df_ratio.copy()
    df_sub_ref_av = df_sub_ref_av - df_ref_av                                                     # subtract this average to the vector
    df_change_per = (df_sub_ref_av/abs(df_ref_av))*100                                            # compute the percent change

    # Exclude from the dataset the reference period:
    df_change_per = df_change_per.drop(df_change_per.index[idx_vector[0]:idx_vector[1]])
    
    ### PERCENTAGE CHANGE AT A SPECIFIC DATE:
    print('Percentage change at the specified date:', df_change_per.iloc[np.where(df_change_per.index == date_string)])
    
    df_out1 = df_ratio
    df_out2 = df_change_per
    return(df_out1, df_out2)


def makeaplot_nocolsel(df_in, 
                       output_dir,
                       timestart, 
                       timeend,
                       wavelength1,
                       wavelength2,
                       label_name,
                       title,
                       dateformat = None, 
                       locator = None, 
                       xlabel = None, 
                       ylabel = None, 
                       yminlim = None, 
                       ymaxlim = None, 
                       ymajlocator = None, 
                       yminlocator = None):
    '''
    function to get an even plot structure
    :argument df_in: dataframe in input
    :argument output_dir: directory where storing the results
    :argument timestart, timeend: starting and ending date
    :argument wavelength1: wavelength numerator
    :argument wavelength2: wavelength denominator
    :argument label_name: sample name
    :argument title: figure title
    :argument dateformat: format of the dates to display
    :argument locator: format the date axis automatically, per day, etc. (e.g. "mdates.AutoDateLocator()"/ "mdates.DayLocator()") 
    :argument xlabel, ylabel: x- and y-axes label
    :argument yminlim, ymaxlim: y-axis limits
    :argument yminlocator, ymajlocator: y-axis major and minor ticks 
    :return: the time series plot   
    '''   
    factor = .5
    plot_width = 25*factor                                                         # define the plot dimensions in inches (width and height)
    plot_height = 13*factor
    df_in.plot(style = ['o'],#,'+','s','v','p'],                                   # the symbol of the markers  
               ms = 3,                                                             # the size of the markers
               label = label_name                                                  # the label in the legend
               )
    
    if dateformat is None:
        dateformat = '%y-%m-%d'
    if locator is None:
        locator = mdates.AutoDateLocator()
    if xlabel is None:
        xlabel = 'Date [yy-mm-dd]'
    if  ylabel is None:
        ylabel = '{} {} '.format('A' + '$_{' + wavelength1 + '}$' + '/' + 'A' + '$_{' + wavelength2 + '}$', '[abs m$^{-1}$]')         
    if yminlim is None:
        ymin = df_in.min()
        yminlim = ymin - 0.05*abs(ymin)
    if ymaxlim is None:
        ymax = df_in.max()
        ymaxlim = ymax + 0.05*abs(ymax) 
    if ymajlocator is None:
        ymajlocator = round((ymaxlim-yminlim)/6,2) 
    if yminlocator is None:
        yminlocator = ymajlocator/3
    
    plt.style.use ('tableau-colorblind10')                                           # use colors colorblind friendly
    plt.xlabel(xlabel)                                                               # label the x-axis 
    plt.ylabel(ylabel)                                                               # label the y-axis 
    plt.rc('font', family='Arial')                                                   # define the font
    plt.rcParams['axes.labelsize'] = 20                                              # define the label size
    #plt.rcParams['axes.labelweight'] = 'bold'                                       # define the label weight
    plt.gca().tick_params(axis = 'both', which = 'major', length = 5, labelsize = 18)# define the label size of the labels on the axes-ticks (major and minor) and length and color of ticks 
    plt.gca().tick_params(which = 'minor', length = 4, colors = 'k')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(dateformat))             # define the format of the date
    plt.gca().xaxis.set_major_locator(locator)                                        # e.g. "AutoDateLocator()"/ "DayLocator()"
    #plt.minorticks_on()                                                              # turn on the minor ticks on both the axes
    plt.gca().set_xlim(timestart,timeend)                                             # define the x-axis range 
    plt.gca().set_ylim(yminlim,ymaxlim)                                               # define the y-axis range
    plt.gca().yaxis.set_major_locator(MultipleLocator(ymajlocator))                   # define the y-axes major and minor ticks 
    plt.gca().yaxis.set_minor_locator(MultipleLocator(yminlocator))                     

    #plt.grid(which='major', linestyle=':', linewidth='1', color='gray')              # define the major grid
    #plt.grid(which='minor', linestyle=':', axis = 'x', linewidth='1', color='gray')  # define the minor grid

    plt.legend(loc='best', fontsize = 17, markerscale = 2)                            # define the legend 
                                                                                      
    plt.rcParams['figure.figsize'] = (plot_width, plot_height)                        # define the size of the figure
    plt.tight_layout() 
    fig = plt.gca().get_figure()                                                      # return the figure 
    fig.savefig(output_dir + title + fig_format, dpi = dpi)                           # save the figure to the desired format and resolution
    
    
def abs_fit_exponential_plot(df_in, 
                             output_dir,                              
                             dateparsingformat, 
                             date_string, 
                             startwl, 
                             endwl,
                             wl0, 
                             xlabel = None, 
                             ylabel = None, 
                             yminlim = None, 
                             ymaxlim = None, 
                             xmajlocator = None, 
                             xminlocator = None,                                
                             ymajlocator = None, 
                             yminlocator = None):  
    '''
    function to fit an exponential curve to the absorbance data at a specific date
    :argument df_in: dataframe in input
    :argument output_dir: directory where storing the results
    :argument dateparsingformat: format of the dates 
    :argument date_string: date to plot
    :argument startwl: starting wavelength
    :argument endwl: ending wavelength
    :argument wl0: reference wavelength in the equation a0*exp(-S*(x-wl0))+K
    :argument xlabel, ylabel: x- and y-axes label
    :argument yminlim, ymaxlim: y-axis limits
    :argument xminlocator, xmajlocator: x-axis major and minor ticks
    :argument yminlocator, ymajlocator: y-axis major and minor ticks 
    :return: the exponential fit (a0*exp(-S*(x-wl0))+K)
    '''   
    factor = .5
    plot_width = 25*factor                                              # inches; define the plot dimensions (width and height)
    plot_height = 13*factor                               

    #1. Find out the best exponential fit to the data
    df_exp = df_in.copy()
    df_exp = df_exp.reset_index()
    df_exp = df_exp.T                                                    # transpose the data frame  
    df_exp = df_exp.reset_index()                                        # restore the original index as row
    headers = df_exp.iloc[0]                                             # rename the dates as headers                                     
    df_exp = pd.DataFrame(df_exp.values[1:], columns = headers)          # convert the first row to header
    df_exp = df_exp.rename(columns={"Timestamp": "wl"})                  # rename the first column as "wl" = wavelength vector
    df_exp['wl'] = df_exp['wl'].replace({'nm':''}, regex=True)           # remove "nm" from the wavelength vector
    wl = df_exp['wl'].apply(pd.to_numeric)                               # convert the wavelength vector to numeric  

    idx = np.where(df_exp.columns == dt.strptime(date_string, dateparsingformat))[0] [0] # get the column index of the date we want to plot
    absorbance = df_exp.iloc[:, idx]                                      # the absorbance vector of the specified date
    sf = interpolate.interp1d(wl, absorbance)
    a0=sf(wl0)
    x = wl[(wl >= startwl) & (wl <= endwl)]  
    y = absorbance[(wl >= startwl) & (wl <= endwl)]
    par_init_vals=[a0,0.02,0.01]                                          # for [a0,S,K]
    par_bounds= ([0, 0, -Inf], [max(y), 1, Inf])                          # for [a0,S,K]
    def cdom_exponential(x, a0, S, K):                                    # define the cdom_exponential function
        return a0*exp(-S*(x-wl0))+K                                       # a*np.exp(-c*(x-b))+d
    popt,pcov=sp.optimize.curve_fit(lambda x,a0,S,K: a0*exp(-S*(x-wl0))+K,  x, y, p0 = par_init_vals, bounds = par_bounds, maxfev = 600) # scipy.optimize.curve_fit uses non-linear least squares to fit a function, f, to data; bounds to constrain the optimization to the region indicated by the lower and upper limit of the a0, S, K parameters; maxfev specifies how many times the parameters for the model that we are trying to fit are allowed to be altered
    print('popt a0,S,K:',popt)                                            # the best-fit parameters: optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized
    a0=round(popt[0],3)                                                   # round the parameter to three digits
    S=round(popt[1],3)
    K=round(popt[2],3)
    print('pcov',pcov)                                                    # the estimated covariance of popt. The diagonals provide the variance of the parameter estimate. 
    perr = np.sqrt(np.diag(pcov))                                         # compute one standard deviation errors on the parameters
    print('perr',perr)

    #2. Plot the exponential fit
    
    if xlabel is None:
        xlabel = 'Wavelength [nm]'
    if  ylabel is None:
        ylabel = 'Absorbance' #[m $^{-1}$]'         
    if yminlim is None:
        ymin = df_in.iloc[:, 1:].values.min()
        yminlim = ymin - 0.05*abs(ymin)
    if ymaxlim is None:
        ymax = df_in.iloc[:, 1:].values.max()
        ymaxlim = ymax + 0.05*abs(ymax) 
    if xmajlocator is None:
        xmajlocator = round((endwl-startwl)/6,0) 
    if xminlocator is None:
        xminlocator = xmajlocator/3
    if ymajlocator is None:
        ymajlocator = round((ymaxlim-yminlim)/6,0) 
    if yminlocator is None:
        yminlocator = ymajlocator/3
 
    start_end_wl=linspace(startwl,endwl,1000)                              # start, end of the interval, the number of items to generate within the range

    plt.style.use ('tableau-colorblind10')                                 # use colors colorblind friendly

    plot(start_end_wl,cdom_exponential(start_end_wl,*popt),                # plot the exponential fit 
        color='k', #black
        #label='fit',
        Linestyle=':', 
        #marker= 'o',
        markersize=2,
        label=f'{a0} $\cdot$ exp(-{S} $\cdot$ ($\lambda$-{wl0}))+ {K}'     # equation of the fit
        )

    #3. Plot the values from the data frame
    plot(x,y, 
        #color='k', 
        label= 'SW',
        Linestyle='None', 
        marker= 'o',
        markersize=2             
        ) 

    plt.style.use ('tableau-colorblind10')                                 # use colors colorblind friendly
    plt.xlabel(xlabel)                                                     # label the x-axis 
    plt.ylabel(ylabel)                                                     # label the y-axis 
    plt.rc('font', family='Arial')                                         # define the font
    plt.rcParams['axes.labelsize'] = 20                                    # define the label size
    #plt.rcParams['axes.labelweight'] = 'bold'                             # define the label weight
    plt.gca().tick_params(axis = 'both', which = 'major', length = 5, labelsize = 18) # define the label size of the labels on the axes-ticks (major and minor) and length and color of ticks 
    plt.gca().tick_params(which = 'minor', length = 4, colors = 'k')
    #plt.minorticks_on()                                                   # turn on the minor ticks on both the axes
    plt.gca().set_ylim(yminlim,ymaxlim)                                    # define the y-axis range
    plt.gca().xaxis.set_major_locator(MultipleLocator(xmajlocator))        # define the y-axes major and minor ticks 
    plt.gca().xaxis.set_minor_locator(MultipleLocator(xminlocator))        # as multiples of a number
    plt.gca().yaxis.set_major_locator(MultipleLocator(ymajlocator))        # define the y-axes major and minor ticks 
    plt.gca().yaxis.set_minor_locator(MultipleLocator(yminlocator))                    

    #plt.grid(which='major', linestyle=':', linewidth='1', color='gray')   # define the major grid
    #plt.grid(which='minor', linestyle=':', axis = 'x', linewidth='1', color='gray')  # define the minor grid

    plt.legend(loc='best', fontsize = 17, markerscale = 2)                 # define the legend 

    plt.rcParams['figure.figsize'] = (plot_width, plot_height)             # define the size of the figure
    plt.tight_layout() 
    fig = plt.gca().get_figure()                                           # return the figure 
    fig.savefig(output_dir + 'exponential_fit_' + str(idx) + fig_format, dpi = dpi) # save the figure to the desired format and resolution
    
    
def abs_fit_exponential(df_in, 
                        startwl,
                        endwl,
                        wl0,
                        sampling_frequency): 

    '''
    function to get a dataframe of optimised parameters and one standard deviation error when fitting an exponential curve to a time series of absorbance data
    :argument df_in: dataframe in input
    :argument startwl: starting wavelength
    :argument endwl: ending wavelength
    :argument wl0: reference wavelength in the equation a0*exp(-S*(x-wl0))+K
    :argument sampling_frequency: sampling frequency
    :return: dataframe with optimised parameters (df_out1) and with one standard deviation error (df_out2)
    '''
    
    df_exp = df_in.copy()
    df_exp = df_exp.reset_index()
    df_exp = df_exp.T                                                 # transpose the data frame  
    df_exp = df_exp.reset_index()                                     # restore the original index as row
    headers = df_exp.iloc[0]                                          # rename the dates as headers                                     
    df_exp = pd.DataFrame(df_exp.values[1:], columns = headers)       # convert the first row to header
    df_exp = df_exp.rename(columns={"Timestamp": "wl"})               # rename the first column as "wl" = wavelength vector
    df_exp['wl'] = df_exp['wl'].replace({'nm':''}, regex=True)        # remove "nm" from the wavelength vector
    wl = df_exp['wl'].apply(pd.to_numeric)                            # convert the wavelength vector to numeric  

    iteration=len(df_exp.columns)                                     # number of loop iterations
    print('number of iterations',iteration)
    empty_matrix=np.zeros((iteration, 3))                             # create an empty matrix to fill in with the parameters of the exponential fit for each datetime  (a0*np.exp(-S*(x-wl0))+K)
    empty_matrix_2=np.zeros((iteration, 3))                           # create an empty matrix to fill in with the 1 std for the parameters of the exponential fit for each datetime
     
    counter=0
    for i in range(0,iteration, sampling_frequency): 
        counter = i
        print(counter) 
        absorbance = df_exp.iloc[:,i]
        x = wl[(wl >= startwl) & (wl <= endwl)]
        y = absorbance[(wl >= startwl) & (wl <= endwl)]
        sf = interpolate.interp1d(wl, absorbance)
        a0=sf(wl0)
        par_init_vals=[a0,0.02,0.01]                                  # for [a0,S,K]
        par_bounds= ([0, 0, -Inf], [max(y), 1, Inf])                  # for [a0,S,K]

        #1. Find out the best exponential fit to the data 
        popt,pcov=sp.optimize.curve_fit(lambda x,a0,S,K: a0*exp(-S*(x-wl0))+K,  x, y, p0 = par_init_vals, bounds = par_bounds, maxfev = 600) # scipy.optimize.curve_fit uses non-linear least squares to fit a function, f, to data; bounds to constrain the optimization to the region indicated by the lower and upper limit of the a0, S, K parameters; maxfev specifies how many times the parameters for the model that we are trying to fit are allowed to be altered
        #print('popt a0,S,K:',popt)                                   # the best-fit parameters.
        a0=round(popt[0],3)                                           # round the parameter to three digits
        S=round(popt[1],3)
        K=round(popt[2],3)    
        empty_matrix[i, :] = popt
        #print(empty_matrix)
        #print('pcov',pcov)                                           # the estimated covariance of popt. The diagonals provide the variance of the parameter estimate. 
        perr = np.sqrt(np.diag(pcov))                                 # compute one standard deviation errors on the parameters
        #print('perr a0,S,K:',perr)
        a0_std=round(perr[0],3)                                       # round the parameter to three digits
        S_std=round(perr[1],3)
        K_std=round(perr[2],3)
        empty_matrix_2[i, :] = perr
        #print(empty_matrix_2)

    names = [_ for _ in 'aSK']
    exp_par = pd.DataFrame(empty_matrix, index=headers, columns=names) # all the headers, excluding the "wl" header
    #print(exp_par)
    exp_par_no_zero = exp_par[exp_par['S'] != 0]                       # keep only rows in which S is different from zero 
    print('Optimised parameters:', '\n', exp_par_no_zero)
    
    names_std = [_ for _ in 'aSK']
    exp_par_std = pd.DataFrame(empty_matrix_2,index=headers,columns=names_std) 
    #print(exp_par_std)
    exp_par_std_no_zero = exp_par_std[exp_par_std['S'] != 0]           # keep only rows in which S is different from zero 
    print('One standard deviation errors on parameters:', '\n', exp_par_std_no_zero)
    
    df_out1 = exp_par_no_zero
    df_out2 = exp_par_std_no_zero 
    return(df_out1, df_out2) 


def abs_spectral_curve(df_in, 
                       starting_date,
                       ending_date,
                       dateparsingformat,
                       sampling_frequency,
                       r2): 
    '''
    function to generate the spectral curve (get a dataframe with negative spectral slope values)
    :argument df_in: dataframe in input
    :argument starting_date: starting date
    :argument ending_date: ending date
    :argument dateparsingformat: format of the dates
    :argument sampling_frequency: sampling frequency
    :argument r2: R2 threshold
    :return: a spectral curve dataframe including columns containing only NaN (df_out1) and a spectral curve dataframe without columns containing only NaN (df_out2)
    '''
    timestamp_selected = pd.date_range(start = starting_date, end = ending_date, periods = 2).tolist() # get the index of the rows corresponding to the defined timestamps to slice the dataframe
    idx_vector = []
    df_sc = df_in.copy()
    df_sc = df_sc.reset_index()
    for timestamp in timestamp_selected:
        index_date_time = np.where(df_sc['Timestamp']==timestamp.strftime(dateparsingformat))[0] 
        idx = index_date_time[:][0]
        idx_vector.append(idx)
    ncol_in = idx_vector[0]                                         # specify that the first and second element of idx_vector are the columns where starting and ending slicing, respectively                                 
    ncol_end = idx_vector[1] 
    df_sc = df_sc.T                                                 # transpose the data frame  
    df_sc = df_sc.reset_index()                                     # restore the original index as row
    headers = df_sc.iloc[0]                                         # rename the dates as headers                                     
    df_sc = pd.DataFrame(df_sc.values[1:], columns=headers)         # convert the first row to header
    df_sc = df_sc.rename(columns={"Timestamp": "wl"})               # rename the first column as "wl" = wavelength vector
    df_sc['wl'] = df_sc['wl'].replace({'nm':''},regex=True)         # remove "nm" from the wavelength vector
    wl = df_sc['wl'].apply(pd.to_numeric)                           # convert the wavelength vector to numeric  
    if df_sc.columns.get_loc('wl') == ncol_in:                      # slice the dataframe, keeping the column "wl"
        df_sc = df_sc.iloc[:, ncol_in:ncol_end]
    else:
        df_sc = df_sc.iloc[:, np.r_[0, ncol_in:ncol_end]]                      
    headers = df_sc.columns        
    iteration = len(df_sc.columns)                                  # number of loop iterations
    print('number of iterations',iteration)
    empty_matrix = np.zeros((iteration, 3))                         # create an empty matrix to fill in with the parameters s275_295, s350_400, SR for each datetime

    interval = 21                                                   # the interval used to calculate each slope
    xx = np.arange(min(wl),max(wl),1).astype(int)
    wl_int=[]                                                       # compute number of average wavelength per interval
    for i in range(min(xx),max(xx)-interval): 
        index = (xx >= i) & (xx <= i + interval)
        wl_int.append(i + (interval / 2))
    nwl   = len(wl_int)
    ncol = len(df_sc.columns)
    ndates = ncol_end-ncol_in
    lin_reg = np.zeros((ndates,nwl))
    lin_reg_r2 = np.zeros((ndates,nwl))
    i=0
    for i in range(0,ndates, sampling_frequency):  
        print(i)
        absorbance = df_sc.iloc[:,i]                                # the absorbance vector
        #Resample data by 1 nm increment:
        xx = np.arange(min(wl),max(wl),1).astype(int)
        sf = interpolate.interp1d(wl, absorbance,kind='cubic')      # spline interpolation of third order
        yy = sf(xx)
        if min(yy)<0:    
            yy = yy-min(yy)
        xx_resh = xx.reshape((-1, 1))
        yy_log_resh = np.log(yy).reshape((-1, 1))
        yy_log_resh[yy_log_resh == float('-inf')]= np.NaN           # to replace -inf with NaN; in R "NA" says no result was available or the result is missing and it can be used in a matrix to fill in a value of a vector
        np.isnan(yy_log_resh)                                       # to see if there are NaN values
        yy=yy_log_resh[np.logical_not(np.isnan(yy_log_resh))].reshape((-1, 1)) # to remove (x, y) pairs where y is nan
        xx=xx_resh[np.logical_not(np.isnan(yy_log_resh))]
        for k, current_wl in enumerate(wl_int):             
                index = (xx >= current_wl - interval/2) & (xx <= current_wl + interval/2)  
                fit=LinearRegression().fit(xx[index].reshape((-1, 1)), yy[index]) 
                lin_reg[i,k]=-fit.coef_[0]
                lin_reg_r2[i,k] = fit.score(xx[index].reshape((-1, 1)),yy[index])
    colnames = [ str(col) for col in wl_int ]
    sc_data = pd.DataFrame(lin_reg, index=headers[int(ncol_in/(ncol_in-1)):int(ncol_end-1/ncol_end)+(ndates-1)], columns=colnames) 
    sc_data_no_zero = sc_data.loc[(sc_data!=0).any(axis=1)]
    sc_r2   = pd.DataFrame(lin_reg_r2, index=headers[int(ncol_in/(ncol_in-1)):int(ncol_end-1/ncol_end)+(ndates-1)], columns=colnames)  
    sc_r2_no_zero = sc_r2.loc[(sc_r2!=0).any(axis=1)] 
    sc_data_r2 = sc_data_no_zero[sc_r2_no_zero>=r2threshold]         # to filter the data and keep only regression with r2 >= r2threshold
    sc_data_r2 = sc_data_r2.dropna(axis=1, how='all')                # to drop the columns with only missing values
    sc_data_r2_no_NaN = sc_data_r2.dropna(axis=1, how='any')         # to drop the columns with only missing values
    sc_data_r2 = sc_data_r2.iloc[1:]
    sc_data_r2.index = pd.to_datetime(sc_data_r2.index)              # make sure time column (here index) is using time format    
    sc_data_r2.index = sc_data_r2.index.rename('Timestamp')          # rename the index column
    sc_data_r2_no_NaN = sc_data_r2_no_NaN.iloc[1:]
    sc_data_r2_no_NaN.index = pd.to_datetime(sc_data_r2_no_NaN.index)# make sure time column (here index) is using time format   
    sc_data_r2_no_NaN.index = sc_data_r2_no_NaN.index.rename('Timestamp') 
    
    df_out1 = sc_data_r2
    df_out2 = sc_data_r2_no_NaN 
    return(df_out1, df_out2)  


def abs_spectral_curve_perchanges(df_in,
                                  date_ref_start,
                                  date_ref_end,
                                  dateparsingformat):
    '''
    function to compute the percentage changes of the negative spectral slope values in relation to a reference period
    :argument df_in: dataframe in input
    :argument date_ref_start: starting date of the reference period
    :argument date_ref_end: ending date of the reference period
    :argument dateparsingformat: format of the dates
    return: a dataframe with percentage changes of the negative spectral slope values in relation to a reference period
    '''
    # Define a reference period:
    date_ref_start1 = str(df_in.iloc[df_in.index.get_loc(date_ref_start, method='nearest')].name) # find the closest date to the date_ref_start and the second closest date to the date_ref_end available in the dataframe
    date_ref_end1 = str(df_in.iloc[df_in.index.get_loc(date_ref_end, method='nearest')+1].name)
    timestamp_selected = pd.date_range(start = date_ref_start1, end = date_ref_end1, periods = 2).tolist()
    idx_vector = []
    for timestamp in timestamp_selected:
            index_date_time=np.where(df_in.index == timestamp.strftime(dateparsingformat))[0] # to check for specific date-time
            idx=index_date_time[:][0]
            idx_vector.append(idx)
    #print('idx_vector:',idx_vector)

    df_ref = df_in.copy()
    df_ref = df_ref[idx_vector[0]:idx_vector[1]]

    # Normalize the data by the average of the reference period and multiply by 100 to get %change:
    df_ref_av = df_ref.mean()                          # compute the average of the vector
    df_sub_ref_av = df_in.copy()
    df_sub_ref_av = df_sub_ref_av - df_ref_av          # subtract this average to the vector
    df_change_per = (df_sub_ref_av/abs(df_ref_av))*100 # compute the percent change

    # Exclude from the dataset the reference period:
    df_change_per = df_change_per.drop(df_change_per.index[idx_vector[0]:idx_vector[1]])
    df_out = df_change_per
    return(df_out)