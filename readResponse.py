#############################################################
#	Compute natural, damped frequencies using the       #
#       logarithmic decrement method                        #
#	Authors: Srinivasa B. Ramisetti  	            #
#	Created:   04-November-2020		            #
#	E-mail: ramisettisrinivas@yahoo.com		    #
#	Web:	http://ramisetti.github.io		    #
#############################################################
#!/usr/bin/env python

import os,sys,argparse,time
import pandas as pd
import dask.dataframe
import numpy as np
import matplotlib.pyplot as plt
from argparse import RawTextHelpFormatter
from dampingEstimateTools import *
#from TopoSignProcess import *

parser = argparse.ArgumentParser(description='Compute damped and natural frequencies using well-established techniques, such as logarithmic decrement method, Hilbert Transform, and Linear Fitting, of response time-series data obtained from OpenFAST and OpenFOAM.\n\
Tested with Python 3.9.0',formatter_class=RawTextHelpFormatter)
parser.add_argument('-f', '--file', nargs='+', required=True, help = "Can accept one or more input data files. Atleast one file should be mentioned")
#parser.add_argument('-s', '--separator', nargs='?', type=str, default='\t', help = "(default: %(default)s), Tab separator is default")
parser.add_argument('-c', '--column', nargs='+', type=str, default='Wave1Elev', help = "(default: %(default)s), Data column/field name")
parser.add_argument('-r', '--nrowsIgn', nargs='?', type=int, default='6', help = "(default: %(default)s), Number of rows to ignore")
parser.add_argument('-tL', '--tMin', nargs='?', type=float, default='0.0', help = "(default: %(default)s), minimum time")
parser.add_argument('-tH', '--tMax', nargs='?', type=float, default='10.0', help = "(default: %(default)s), maximum time")
parser.add_argument('-dR', '--dampRatio', nargs='?', type=float, default=None, help = "(default: %(default)s), Damping ratio for Logarithmic decrement method")
parser.add_argument('-adv', '--advMode', default=False, action='store_true')
args = parser.parse_args()
print(args)

tL=args.tMin
tH=args.tMax
dampRatio=args.dampRatio
nrowsIgnore=args.nrowsIgn;
#column_name=args.column
listOfData={}
filenames=sys.argv[1:]
args.column.append('Time') # this is necessary to filter rows

def get_counts(df, usecols):
    by_party = df.groupby("Time")
    street = by_party[usecols]
    return street.value_counts()

for i in args.file:
    filename=str(i)
 
    s = time.time()
    # skip the first nrowsIgnore lines and read the data into df
    df = pd.read_csv(filename, sep='\s+', skiprows=nrowsIgnore, low_memory=False, encoding="utf-8-sig", skipinitialspace=True, usecols=args.column, chunksize=50000)
    #df = dask.dataframe.read_csv(filename, sep='\t', skiprows=nrowsIgnore, encoding="utf-8-sig")
    df = pd.concat(df)
    #df = get_counts(df,'RotSpeed')
    e = time.time()
    print("Pandas file reading time = {}".format(e-s))
    #print(df.compute())
    
    # delete the empty spaces within the header names
    df.columns = df.columns.str.strip().str.replace(' ', '')
    
    # skip the next row with units after the header row
    # and create new df with all columns data in float
    new_df=df.iloc[1:].astype(float)

    # copy the time data tL and tH
    nd=new_df[new_df['Time'].between(tL,tH)]

    fieldData={}
    for column_name in args.column:
        if not column_name in new_df.columns and column_name != 'Time':
            print("Column with label name ", column_name, " does not exit! Check the label name.")
            exit()
        else:
            # copy the WaveElev data and append to list
            fieldVals=nd[column_name].to_numpy()
            #fieldData.append(fieldVals)
            fieldData[column_name]=fieldVals
    listOfData[filename]=fieldData;
    

# uncomment for time logging
# s = time.time()

#for id, data in zip(args.column, listOfData):
for field in args.column:
    if field=='Time':
        continue
    #f = open(field+'.dat', "a")
    for cas in args.file:
        x=nd['Time'].to_numpy()
        data=listOfData[cas][field]
        
        xfitpeak,yfitpeak,max_idx,min_idx,decayRate=logarithmicDecrement(x,data,dampRatio)

        plt.figure(1)
        # Plot the response data
        plt.plot(x, data, 'b-', label='Heave response')
        # Add green points at data points preceding an actual zero crossing.
        # plt.plot(x[idx], y[idx], 'gp')
        # Add red points at peak data points.
        plt.plot(x[max_idx], data[max_idx], 'rp')
        plt.plot(x[min_idx], data[min_idx], 'g+')

        if(len(xfitpeak)>=3):
            #p1, pcov1 = curve_fit(funcLinearExp, xfit, yfit, p0=(ypeak[0],1.,1.), maxfev=5000)
            p1, pcov1 = curve_fit(funcLinearExp, xfitpeak, yfitpeak, p0=(yfitpeak[0],0.1,1.), maxfev=5000)
            #p1d, pcov1d = curve_fit(funcLinearExpDown, xfitdown, yfitdown, p0=(yfitdown[0],0.97,1.), maxfev=5000)
            ans1=funcLinearExp(x,p1[0],p1[1],p1[2])
            #ans1d=funcLinearExpDown(x,p1d[0],p1d[1],p1d[2])
            #stddev_p1d = np.sqrt(np.diag(pcov1d))
            #print('p1d ', p1d,'pcov1d ', stddev_p1d)
            
            ansLD=funcLinearExp(x,p1[0],decayRate[0],p1[2])
            plt.plot(x,ansLD,color='pink',label='Logarithmic Decrement Fit', ls=linestyle_tuple['densely dashed'])
            
            #print('p1d ', p1d,'pcov1d ', pcov1d)
            print('----- Parameter Estimation for Linear Damping using Exponential Decay Fit')
            stddev_p1 = np.sqrt(np.diag(pcov1))
            print('p1 ', p1,'pcov1 stddev ', stddev_p1)
            print('Initial Amplitude A0', p1[0], '+-', stddev_p1[0])
            print('Decay Rate (Hz)', p1[1], '+-', stddev_p1[1])
            plt.plot(x,ans1,color='cyan',label='Exponential Decay Fit', ls=linestyle_tuple['dotted'])     
          
            # print('----- Parameter Estimation for Linear Damping (Troughs)')
            # print('Initial Amplitude A0', p1d[0])
            # print('Decay Rate ', p1d[1])

            #ansLD=funcLinearExp(x,p1[0],decayRate[0],p1[2])
            #plt.plot(x,ansLD,color='cyan',label='Logarithmic Decrement Fit', ls=linestyle_tuple['densely dashed'])


            if args.advMode:
                # apply performLinearQuadraticFit
                performLinearQuadraticFit(x,xfitpeak,yfitpeak)
    
        plt.xlabel('Time (s)')
        plt.ylabel('Heave (-)')
        plt.legend()
        plt.show()
        
        if args.advMode:
            # use hilbert transform method to estimate decay rate
            envelope=performHilbertTransform(x,data)
            plt.figure(2)
            plt.subplot(1, 2, 1)
            plt.plot(x,data,color='b',label='Heave response')
            plt.plot(x,envelope, color='r', label='Hilbert Envelope')
            plt.xlabel('Time (s)')
            plt.ylabel('Decay response (-)')
            plt.legend()
            plt.subplot(1, 2, 2)
            lInd=1
            hInd=len(x)-1
            p1, pcov1 = curve_fit(funcLinearExp, x[lInd:hInd], envelope[lInd:hInd], p0=(envelope[lInd],1.,1.), maxfev=5000)
            ans1=funcLinearExp(x,p1[0],p1[1],p1[2])
            print('----- Damping Estimation using Hilbert Transform')
            stddev_p1 = np.sqrt(np.diag(pcov1))
            print('Decay Rate ', p1[1], '+-', stddev_p1[1])
            plt.plot(x,envelope, color='b', label='Hilbert Envelope')
            #fitlabel = r'Linear fit, $n$ = {} $\times$ 10$^{{{}}}$ cm$^{{-3}}$'.format(c1, c2)
            plt.plot(x,ans1, color='r', label='Fit: (%5.2e +- %5.3f)' % (p1[1],stddev_p1[1]))
            plt.yscale('log')
            plt.xlabel('Time (s)')
            plt.ylabel('Decay response (-)')
            plt.legend()
            plt.tight_layout()
            plt.show()

            # use periodgram to estimate damped frequency
            periodogram,freqs,kmax=applyPeriodogram(x,data)
            plt.figure(3)
            plt.plot(freqs, np.sqrt(4*periodogram/(10*len(data))))
            plt.xlabel('Frequency (rad/s)')
            plt.grid()
            plt.axvline(freqs[kmax], color='r', alpha=0.25, label='Damped Frequency (rad/s)')
            plt.legend()
            plt.show()

            ## use 0D persistence diagram for damping estimation
            # damping = 'viscous'
            # Result = damping_constant(x, data, damping, plotting = True)

            # SigFigs = 4
            # print(Result['floor'])
            # print(Result.keys())
            # print('zeta (opt, fit, one): ', 
            #       round(Result['damping_params']['zeta_opt'],SigFigs),
            #       round(Result['damping_params']['zeta_fit'],SigFigs),
            #       round(Result['damping_params']['zeta_one'],SigFigs))
            # print('mu (opt, fit, one):   ', 
            #       round(Result['damping_params']['mu_opt'],SigFigs),
            #       round(Result['damping_params']['mu_fit'],SigFigs),
            #       round(Result['damping_params']['mu_one'],SigFigs))

        
        #id=os.path.basename(filename)
        #maxX=np.max(data)
        #minX=np.min(data)
        #v,blockVar,blockMean=BlkAvg.blockAverage(data,False,blkSize)
        #print(field, cas, blockMean[-1], np.sqrt(blockVar[-1]), blockVar[-1], minX, maxX)
        #print("<x> = {0:e} +/- std: {1:e} var: {1:e} minX: {1:e} maxX: {1:e}\n".format(blockMean[-1], np.sqrt(blockVar[-1]), blockVar[-1], minX, maxX))
        #f.write('%s %g %g %g %g %g\n' %(cas, blockMean[-1], np.sqrt(blockVar[-1]), blockVar[-1], minX, maxX))
    #f.close()


# e = time.time()
# print("Time spent in logarithmic decrement function = {}".format(e-s))
