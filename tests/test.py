# References:OA
# 1. https://en.wikipedia.org/wiki/Logarithmic_decrement
# 2. https://www.brown.edu/Departments/Engineering/Courses/En4/Notes/vibrations_free_damped/vibrations_free_damped.htm
# 3. https://www.osti.gov/servlets/purl/1141201

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt
#import scipy.signal as sig
#from scipy.signal import lombscargle
#from collections import OrderedDict
from scipy.signal import hilbert, savgol_filter

sys.path.append('../')

from dampingEstimateTools import *
from TopoSignProcess import *

if __name__ == "__main__":

    # create known data signal
    x = np.linspace(0,200,num=10000)
    y = 2 * np.cos(2*np.pi*x/10 - 0.25) * np.exp(-x*0.02)

    # use logarithmic decrement method
    xfitpeak,yfitpeak,max_idx,min_idx,decayRate=logarithmicDecrement(x,y)

    plt.figure(1)
    # Plot the data
    plt.plot(x, y, 'b-', label='Heave response')
    # Add green points at data points preceding an actual zero crossing.
    # plt.plot(x[idx], y[idx], 'gp')
    # Add red points at peak data points.
    plt.plot(x[max_idx], y[max_idx], 'rp')
    plt.plot(x[min_idx], y[min_idx], 'g+')

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
        print('----- Parameter Estimation for Linear Damping using Exponential Decay Fit ')
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


        performLinearQuadraticFit(x,xfitpeak,yfitpeak)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Heave (-)')
    plt.legend()
    plt.show()
    
    # use hilbert transform method to estimate decay rate
    envelope=performHilbertTransform(x,y)
    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.plot(x,y,color='b',label='Heave response')
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

    exit()
    # use periodgram to estimate damped frequency
    periodogram,freqs,kmax=applyPeriodogram(x,y)
    plt.figure(3)
    plt.plot(freqs, np.sqrt(4*periodogram/(10*len(x))))
    print('----- Damped Frequency Estimation using DFT')
    plt.xlabel('Frequency (rad/s)')
    plt.grid()
    plt.axvline(freqs[kmax], color='r', alpha=0.25, label='Damped Frequency (rad/s)')
    plt.legend()
    plt.show()

    # use sublevel persistence diagrams to estimate decay rate
    damping = 'viscous'
    #Result = damping_constant(x, y, damping, plotting = True)
    exit()
