# References:OA
# 1. https://en.wikipedia.org/wiki/Logarithmic_decrement
# 2. https://www.brown.edu/Departments/Engineering/Courses/En4/Notes/vibrations_free_damped/vibrations_free_damped.htm
# 3. https://www.osti.gov/servlets/purl/1141201

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt
import scipy.signal as sig
from scipy.signal import lombscargle
from collections import OrderedDict
from scipy.signal import hilbert, savgol_filter
#from TopoSignProcess import *

linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = OrderedDict([
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])


# ==== fitting using leastsq ====
def residuals(args, t, x):
    return x - decay_cosine(t, *args)

# ==== Linear function  ====
def funcLinear(x, a, b):
     return a * x + b
 
# ==== decay cosine function  ====
def decay_cosine(x, a, b, c, d):
     return (a * np.cos(2*np.pi*x*b-c)) * np.exp(-d*x)

# ==== decay function  ====
def funcLinearExp(x, a, b, c):
     return a * np.exp(-b*x)+c

# ==== growth function  ====
def funcLinearExpDown(x, a, b, c):
     return a * (1-np.exp(-b*x))+c

# ==== decay cosine function with linear+quadratic terms  ====
def funcLinearQuadExp(x, a, b, c, d):
     return a * b * np.exp(-b*x)/(b+a*c*(1-np.exp(-b*x)))+d
 
# use logarithmic decrement and linear/quadratic fitting to estimate damping and natural frequency
def logarithmicDecrement(x,y,dampRatio=None):
     zerocross=0.0
     zerocross_tol=5e-2

     #apply a Savitzky-Golay filter
     #y = savgol_filter(y, window_length = 351, polyorder = 5)
     
     # Get coordinates and indices of zero crossings
     idx = np.where(np.diff(np.sign(y)))[0]
     print('Zero crossing indices: ', idx)

     # Find Positive peak indices
     # avoid using find_peaks as it not picking the first max element
     # peak_idx,_=sig.find_peaks(y,height=0.0001,width=4)
     # print(max_idx)
     tmp_id=sig.argrelextrema(np.array(y),np.greater_equal)
     max_idx=np.asarray(tmp_id)[0]
     # print(type(max_idx))
     #exit()
     #height = max_idx[1]['peak_heights'] #list containing the height of the peaks

     ht=y[max_idx]
     max_idx=max_idx[ht>0] # consider ids whose peaks are greater than 0
     height=y[max_idx]
     max_pos = x[max_idx]   #list containing the positions of the peaks
     print('Positive peak indices: ', max_idx)

     #Find Negative peak indices
     tmp_id=sig.argrelextrema(np.array(y),np.less_equal)
     min_idx=np.asarray(tmp_id)[0]
     #min_idx,_ = sig.find_peaks(y2, threshold = 1, distance = 1)
     ht=y[min_idx]
     min_idx=min_idx[ht<0] # consider ids whose downs are lesser than 0
     min_pos = x[min_idx]   #list containing the positions of the minima
     min_height = y[min_idx]   #list containing the height of the minima
     print('Negative peak indices: ', min_idx)

     # initialize T array with size len(idx)-2 as it requires 3 points to compute the time period
     if(len(idx)<3):
          print('Not enough zero crossings to proceed further!')
          exit()
     T=np.zeros(len(idx)-2) 
     for i in range(len(T)):
          T[i]=x[idx[i+2]]-x[idx[i]]
     print('Time period in seconds based on zero crossings: ',T)

     Zn=np.zeros(len(max_idx))
     for i in range(len(Zn)):
          Zn[i]=y[max_idx[i]]
     print('Peak values: ',Zn)

     ZnD=np.zeros(len(min_idx))
     for i in range(len(ZnD)):
          ZnD[i]=y[min_idx[i]]
     print('Down values: ',ZnD)

      # initialize T_v2 array with size len(idx)-2 as it requires 3 points to compute the time period
     T_P=np.zeros(len(max_idx)-1) 
     for i in range(len(T_P)):
          T_P[i]=x[max_idx[i+1]]-x[max_idx[i]]
     print('Time period in seconds based on peaks: ',T_P)

     T_D=np.zeros(len(min_idx)-1) 
     for i in range(len(T_D)):
          T_D[i]=x[min_idx[i+1]]-x[min_idx[i]]
     print('Time period in seconds based on downs: ',T_D)

     # delta0=np.zeros(len(Zn)-1)
     # for i in range(len(delta0)):
     #      N=i+1;
     #      delta0[i]=np.log(y[max_idx[i]]/y[max_idx[i+1]])/N
     # print('Logarithmic decrement v1: ',delta0)

     delta0=np.zeros(len(Zn)-1)
     for i in range(len(delta0)):
          delta0[i]=np.log(y[max_idx[i]]/y[max_idx[i+1]])
     print('Logarithmic decrement: ',delta0)

     dampratio_v2=np.zeros(len(Zn)-1)
     for i in range(len(dampratio_v2)):
          N=i+1;
          dampratio_v2[i]=delta0[i]/np.sqrt(4*np.pi*np.pi+delta0[i]*delta0[i])
          #print(N)
     print('Damping ratio v2: ',dampratio_v2)

     mean_delta_v1 = delta0[0]; # consider the first two peaks only
     mean_delta_v2 = np.mean(delta0); # consider mean of all delta values
     mean_delta_v3 = np.mean(delta0[0:2]); # consider mean of the first 2 delta values i.e. first 3 peaks only
     #print(dampRatio)
     if dampRatio != None:
         damp_ratio_v1=dampRatio
     else:
         damp_ratio_v1 = delta0[0]/np.sqrt(4*np.pi**2+delta0[0]**2)
         
     damp_ratio_v2 = mean_delta_v2/np.sqrt(4*np.pi**2+mean_delta_v2**2)
     damp_ratio_v3 = mean_delta_v3/np.sqrt(4*np.pi**2+mean_delta_v3**2)
     
     damp_ratio=np.array([damp_ratio_v1, damp_ratio_v2, damp_ratio_v3])
     omega_d=np.array([2*np.pi/(T_P[0]), 2*np.pi/np.mean(T_P), 2*np.pi/np.mean(T_P[0:2])])
     omega_n=np.array([omega_d[0]/np.sqrt(1-damp_ratio[0]**2), omega_d[1]/np.sqrt(1-damp_ratio[1]**2),omega_d[2]/np.sqrt(1-damp_ratio[2]**2)])
     omega_d_Hz=omega_d/2/np.pi
     omega_n_Hz=omega_n/2/np.pi
     decayRate=damp_ratio*omega_n

     print('Mean Delta (-) v1, v2,v3', mean_delta_v1, mean_delta_v2, mean_delta_v3)
     print('Damping Ratio (-) v1,v2,v3',damp_ratio)
     print('Damped Frequency (rad/sec) ',omega_d)
     print('Natural frequency (rad/sec)',omega_n)
     print('Damped Frequency (Hz) ',omega_d_Hz)
     print('Natural frequency (Hz)', omega_n_Hz)
     print('Decay Rate (Hz)', decayRate)

     print('----- Parameter Estimation of Linear Damping, Damped Frequency and Natural Frequency using Logaritmic Decrement Fit ')
     print('Damped Frequency (rad/sec) ',omega_d[0])
     print('Natural frequency (rad/sec)',omega_n[0])
     print('Damped Frequency (Hz) ',omega_d_Hz[0])
     print('Natural frequency (Hz)', omega_n_Hz[0])
     print('Decay Rate (Hz)', decayRate[0])

     # plt.figure(1)
     # # Plot the data
     # plt.plot(x, y, 'b-', label='Heave response')
     # # Add green points at data points preceding an actual zero crossing.
     # # plt.plot(x[idx], y[idx], 'gp')
     # # Add red points at peak data points.
     # plt.plot(x[max_idx], y[max_idx], 'rp')
     # plt.plot(x[min_idx], y[min_idx], 'g+')

     xpeak=x[max_idx];
     ypeak=y[max_idx];
     xdown=x[min_idx];
     ydown=y[min_idx];

     # filter peak and downs to avoid unwanted data points
     xfitpeak=xpeak[ypeak>0]
     yfitpeak=ypeak[ypeak>0]
     xfitdown=xdown[ydown<0]
     yfitdown=ydown[ydown<0]
     
     # #p, pcov = curve_fit(funcExp, xpeak, ypeak, p0=(1,1e-5), maxfev=5000)
     # if(len(xfitpeak)>=3):
     #      #p1, pcov1 = curve_fit(funcLinearExp, xfit, yfit, p0=(ypeak[0],1.,1.), maxfev=5000)
     #      p1, pcov1 = curve_fit(funcLinearExp, xfitpeak, yfitpeak, p0=(yfitpeak[0],0.1,1.), maxfev=5000)
     #      #p1d, pcov1d = curve_fit(funcLinearExpDown, xfitdown, yfitdown, p0=(yfitdown[0],0.97,1.), maxfev=5000)
     #      ans1=funcLinearExp(x,p1[0],p1[1],p1[2])
     #      #ans1d=funcLinearExpDown(x,p1d[0],p1d[1],p1d[2])
     #      #stddev_p1d = np.sqrt(np.diag(pcov1d))
     #      #print('p1d ', p1d,'pcov1d ', stddev_p1d)

     #      ansLD=funcLinearExp(x,p1[0],decayRate[0],p1[2])
     #      plt.plot(x,ansLD,color='pink',label='Logarithmic Decrement Fit', ls=linestyle_tuple['densely dashed'])
          
     #      #print('p1d ', p1d,'pcov1d ', pcov1d)
     #      print('----- Parameter Estimation for Linear Damping ')
     #      stddev_p1 = np.sqrt(np.diag(pcov1))
     #      print('p1 ', p1,'pcov1 stddev ', stddev_p1)
     #      print('Initial Amplitude A0', p1[0], '+-', stddev_p1[0])
     #      print('Decay Rate (Hz)', p1[1], '+-', stddev_p1[1])
     #      plt.plot(x,ans1,color='cyan',label='Exponential Decay Fit', ls=linestyle_tuple['dotted'])     
          
     #      # print('----- Parameter Estimation for Linear Damping (Troughs)')
     #      # print('Initial Amplitude A0', p1d[0])
     #      # print('Decay Rate ', p1d[1])

     #      #ansLD=funcLinearExp(x,p1[0],decayRate[0],p1[2])
     #      #plt.plot(x,ansLD,color='cyan',label='Logarithmic Decrement Fit', ls=linestyle_tuple['densely dashed'])

     # plt.xlabel('Time (s)')
     # plt.ylabel('Heave (-)')
     # plt.legend()
     # plt.show()

     # x0 = np.ones(4)  # initializing all params at one
     # params_lsq, _ = scipy.optimize.leastsq(residuals, x0, args=(x, y))
     # print(params_lsq)


     return xfitpeak,yfitpeak,max_idx,min_idx,decayRate

def performLinearQuadraticFit(x,xfitpeak,yfitpeak):
    if(len(xfitpeak)>=4):
        p2, pcov2 = curve_fit(funcLinearQuadExp, xfitpeak, yfitpeak, p0=(yfitpeak[0],1.,1.,1.), maxfev=5000)
        ans2=funcLinearQuadExp(x,p2[0],p2[1],p2[2],p2[3])
        plt.plot(x,ans2,color='olive',label='Linear-plus-quadratic Fitting', ls=linestyle_tuple['dashdotdotted'])
        print('----- Parameter Estimation for Linear-plus-quadratic Damping using Exponential Decay Fit')
        stddev_p2 = np.sqrt(np.diag(pcov2))
        print('Initial Amplitude A0 ', p2[0], '+-', stddev_p2[0])
        print('Linear Decay Rate ', p2[1], '+-', stddev_p2[1])
        print('Quadratic Decay Rate ', p2[2], '+-', stddev_p2[2])
        plt.legend()


def performDirectDecayCosineFit(x,y):
    if(len(x)>=4):
        p3, pcov3 = curve_fit(decay_cosine, x, y, p0=(y[0],0.63,0.26,1.74), maxfev=50000)
        #p3, pcov3 = curve_fit(decay_cosine, x, y)
        ans3=decay_cosine(x,p3[0],p3[1],p3[2],p3[3])
        print('p3 ', p3,'pcov3 ', pcov3)
        plt.plot(x,ans3,'-',color='yellow',label='Exp*Cos Fitting', linestyle=linestyle_tuple['densely dashed'])
        print('----- Parameter Estimation for Nonlinear A0*Exp(-v*t)*Cos(w_d*t-phase) Damping')
        print('Initial Amplitude A0', p3[0])
        print('Damped Time period (sec) ', p3[1])
        print('Phase (-) ', p3[2])
        print('Decay Rate ', p3[3])
    
def performHilbertTransform(x,y):
     # Using Hilbert Transform to get the decay envelope
     # https://stackoverflow.com/questions/35721100/determining-the-event-location-on-an-oscillatory-signal
     # Add some padding to limit the periodic extension effect
     padlen = int(np.floor(0.1*len(x)))
     ypad = np.pad(y, (0,padlen), mode='edge');
     # Compute the envelope
     envelope = np.abs(hilbert(ypad));
     # Truncate to the original signal length
     envelope = envelope[:-padlen]

     return envelope

def applyPeriodogram(x,y):
     # Using periodogram to extract Damped frequency
     # https://stackoverflow.com/questions/34428886/discrete-fourier-transformation-from-a-list-of-x-y-points
     n=len(x)
     dxmin = np.diff(x).min()
     duration = x.ptp()
     freqs = np.linspace(1/duration, n/duration, 10*n)
     periodogram = lombscargle(x, y, freqs)

     print('----- Damped Frequency Estimation using DFT')
     kmax = periodogram.argmax()
     print("Damped Frequency from DFT: %8.3f (rad/s)" % (freqs[kmax],))
     return periodogram,freqs,kmax

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
