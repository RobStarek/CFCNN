
# In[1]:

import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import h5py

#Get info from generator
from generate_inputs import MONTE_CARLO_SAMPLES, PSFW, RAYLEIGH, WIDTH, HEIGTH, UPSCALE, snrs, distances
snrs_mc = np.kron(snrs,np.ones(MONTE_CARLO_SAMPLES))

key_dict = dict()
counter = 0
for snr in snrs_mc:
    key = f'snr_{int(round(snr)):d}_{counter}' if snr>0 else 'noiseless'
    counter += 1
    key_dict[key] = snr

#Load inputs and outputs
with h5py.File('outputs/cnn_output_res_psf2.h5', 'r') as h5f:
    cnndata = {key : np.array(h5f[key]).reshape((-1,200,200)) for key in h5f.keys()}

with h5py.File('inputs/input_res_for_cnn_psf2.h5', 'r') as h5f:
    inpdata = {key : np.array(h5f[key]).reshape((-1,50,50)) for key in h5f.keys()}
    

# In[18]:
# Define helper functions


def estimate_com(y):    
    """
    Estimate center of mass of peaks in a 1D array.
    Peaks are located by a simple thresholding.
    """
    thr = np.max(y)*0.5
    hi = (y > thr)
    edges = np.diff(hi.astype(float))
    peak_rising = edges > 0.1
    peak_falling = edges < -0.1
    n_rising_edges = np.sum(peak_rising)
    n_falling_edges = np.sum(peak_falling)
    try:
        assert (n_falling_edges == n_rising_edges) or ((n_falling_edges > 1) or (n_rising_edges > 1))
    except AssertionError:
        return list()
    n_peaks = n_rising_edges
    x = np.arange(len(y))
    coms = [
        (x[start:stop+1] @ y[start:stop+1])/np.sum(y[start:stop+1]) 
        for start, stop in 
        zip(np.where(peak_rising)[0], np.where(peak_falling)[0])
        ]
    return coms
        
def is_easialy_resolvable(img):
  """
  Check whether peaks in the middle row of the image are easily resolvable.
  """
  #this is approximation valid only for emitters places in roughly same row of the image
  #I use it because it is easy to do and it is OK for my data, but keep in mind 
  #that this is NOT a general method
  #generally, one would need to use sophisticated 2d peak detection
  xmarginal = np.mean(img, axis=0)
  try:
      coms = estimate_com(xmarginal)
  except AssertionError:
      # plt.matshow(img)
      # plt.show()
      # raise
      return np.nan
  
  n = len(coms)
  if n >= 2:
      if n > 2:
          print(f"Warning: {n} peaks found")
      return True
  return False

def is_somehow_resolvable(signal):
# def isr(s):
    sn = signal/np.max(signal)
    sel = (sn > 0.2).astype(int)    
    areas = []
    rising = np.diff(sel) > 0.1
    falling = np.diff(sel) < -0.1
    rising_count = sum(rising)
    falling_count = sum(rising)
        
    starts = [i for i, r in enumerate(rising) if r]
    stops = [i for i, r in enumerate(falling) if r]
    try:
        peaks = sorted([np.max(sn[i:j]) for i,j in zip(starts, stops)])
    except:
        return False

    if len(peaks) >= 2:
        mean_peak = np.min(peaks[-2:])
    else:
        mean_peak = 1
    hi = sn > mean_peak*(1-0.263)
    dhi = np.diff(hi.astype(np.float32))

  if (rising_count >= 2) and  (falling_count >= 2):
        if rising_count == falling_count:
            return True        
    rising = dhi > 0.1
    falling = dhi < -0.1
    rising_count = sum(rising)
    falling_count = sum(rising)        
    if (rising_count >= 2) and  (rising_count >= 2):
        return True
    return False

# In[19]:

snr_groups = dict()
for key in cnndata:
    if 'snr' not in key:
        continue
    identifier, snrval, trial = key.split('_')
    snrkey = f'{key_dict[key]:.1f}'
    if snrkey in snr_groups:
        snr_groups[snrkey].append(key)
    else:
        snr_groups[snrkey] = [key]
sorted_snr_keys = sorted(snr_groups.keys(), key = lambda x: float(x))
snr_table = np.array(list(map(float, sorted_snr_keys)))
sorted_snr_keys


# In[3]:


def plot_estimate_resolution(snrkey):
    keys = snr_groups.get(snrkey)
    if float(snrkey)<10:
        return None
    
    ones = np.ones(4)
    arr = []
    arr2 = []
    for key in keys:        
        arr.append(np.array([np.mean(im, axis=0) for im in cnndata[key]]))
        arr2.append(np.array([np.kron(np.mean(im, axis=0), ones) for im in inpdata[key]]))
    arr = np.mean(np.array(arr), axis=0)
    arr2 = np.mean(np.array(arr2), axis=0)
    arr /= np.max(arr)
    arr2 /= np.max(arr2)
    resolvable = np.array([is_somehow_resolvable(y) for y in arr])
    inconclusive = False
    if resolvable[0]:
        inconclusive = True
    win = 1
    cres = [np.all(resolvable[i:i+win]) for i in range(len(resolvable)-win)]
    thr = np.where(cres)[0]
    if len(thr) > 0:
        thr = thr[0]
    else:
        thr = 0
    
    composite = np.zeros((41,200,3))
    composite[:,:,0] = arr**0.5
    composite[:,:,1] = arr2
    plt.imshow(composite)
    plt.title(snrkey)
    if not inconclusive:
        plt.hlines(thr, 0, 199, colors='r')
        plt.annotate(f'{distances[thr]/RAYLEIGH:.2f} R.d.', (20, thr-2), c='r')
        plt.show()
        plt.plot(arr[0]/np.max(arr[0]), c='k')
        plt.plot(arr[thr]/np.max(arr[thr]))
        plt.xlim(80,120)
        if thr > 1:
            plt.plot(arr[thr-1]/np.max(arr[thr-1]))
        plt.show()
    else:
        plt.title(snrkey+'???!')
        plt.show()
        plt.plot(arr[0])
        plt.show()
    plt.show()

for skey in sorted_snr_keys:
    plot_estimate_resolution(skey)


# In[4]:


def estimate_resolution(snrkey):
    keys = snr_groups.get(snrkey)
    ones = np.ones(4)
    arr = []    
    for key in keys:
        estims = []
        for dframe in cnndata[key]:
            marginal = np.mean(dframe, axis=0)
            marginal /= np.max(marginal)
            resolvable = is_somehow_resolvable(marginal)
            estims.append(resolvable)
        estims = np.array(estims)
        win = 1
        filtered_resolvability = [np.all(estims[i:i+win]) for i in range(len(estims)-win)]
        conclusive = (True in filtered_resolvability)
        if not conclusive:
            continue
        ires = np.where(filtered_resolvability)[0][0]
        if conclusive:
            arr.append(ires)
    succ_fraction = len(arr)/10
    return np.mean(distances[arr])/RAYLEIGH, np.std(distances[arr])/RAYLEIGH, succ_fraction

evaluated_performance = []
for snrval, skey in zip(snr_table, sorted_snr_keys):
    x, dx, s = estimate_resolution(skey)
    evaluated_performance.append((snrval, x, dx, s))
    print(f'{snrval:.1f} {x:.02f}+/-/{dx:.02f}, {s*100:.0f}%')

evaluated_performance = np.array(evaluated_performance)


# In[22]:


plt.semilogx(evaluated_performance[:,0],evaluated_performance[:,1], "o-", label='minimal resolvable distance')
plt.semilogx(evaluated_performance[:,0],evaluated_performance[:,2], "s-", label='conclusiveness')
plt.legend()
plt.grid()
plt.xlabel('SNR')

# save
with h5py.File('results_psf2.h5', 'w') as h5f:
    dset = h5f.create_dataset('table', data = evaluated_performance)
    dset.attrs['info'] = ('snr', 'rel_resolution_mean', 'rel_resolution_std', 'conclusiveness')
    
# plt.ylabel('resolvable distance [Rayleigh limit]')


# In[5]:


for i in range(10):
    plt.plot(np.mean(cnndata[f'snr_1000_{360+i}'][0], axis=0), c='k', alpha=0.5)
for i in range(10):
    plt.plot(np.mean(cnndata[f'snr_1000_{360+i}'][12], axis=0), c='r', alpha=0.5)    
plt.xlim(75,125)
plt.show()


# In[6]:


with h5py.File('inputs/res_rl_psf2.h5', 'r') as h5f:
    rldata = {key : np.array(h5f[key]).reshape((-1,200,200)) for key in h5f.keys()}


# In[10]:


rldata.keys()


# In[11]:


def is_somehow_resolvable_rl(signal):
# def isr(s):snr_10_121
    sn = signal/np.max(signal)
    sel = (sn > 0.3).astype(int) #RL act nicely in terms of peak height, but have non-negligable background which needs to be addressed by higher thresholds
    areas = []
    rising = np.diff(sel) > 0.1
    falling = np.diff(sel) < -0.1
    rising_count = sum(rising)
    falling_count = sum(rising)
        
    starts = [i for i, r in enumerate(rising) if r]
    stops = [i for i, r in enumerate(falling) if r]
    try:
        peaks = sorted([np.max(sn[i:j]) for i,j in zip(starts, stops)])
    except:
        return False

    #mean peak approach
    # if len(peaks) >= 2:
    #     mean_peak = np.mean(peaks[-2:])
    # else:
    #     mean_peak = 1
    #min peak approach
    if len(peaks) >= 2:
        mean_peak = np.min(peaks[-2:])
    else:
        mean_peak = 1
    hi = sn > mean_peak*(1-0.263)
    dhi = np.diff(hi.astype(np.float32))
    # print(mean_peak, mean_peak*(1-0.263), peaks)
    # plt.plot(sn)
    # plt.plot(sel)
    # plt.hlines(peaks,80,120, colors='k')
    # plt.hlines([mean_peak],80,120, colors='g')
    # plt.hlines([mean_peak*(1-0.263)],80,120, colors='r')
    # plt.xlim(80,120)
    if (rising_count >= 2) and  (falling_count >= 2):
        if rising_count == falling_count:
            return True     
    rising = dhi > 0.1
    falling = dhi < -0.1
    rising_count = sum(rising)
    falling_count = sum(rising)        
    if (rising_count >= 2) and  (rising_count >= 2):
        return True
    return False
    
snr_groups = dict()
for key in rldata:
    if 'snr' not in key:
        continue
    identifier, snrval, trial = key.split('_')
    snrkey = f'{key_dict[key]:.1f}'
    if snrkey in snr_groups:
        snr_groups[snrkey].append(key)
    else:
        snr_groups[snrkey] = [key]
sorted_snr_keys = sorted(snr_groups.keys(), key = lambda x: float(x))
snr_table = np.array(list(map(float, sorted_snr_keys)))
print(sorted_snr_keys)

def rl_bckg_sub(y):    
    bckg_est = np.quantile(y, 0.8)
    y[0:50] = 0
    y[-50:] = 0
    return np.clip(y - bckg_est, 0, None)
    
def plot_estimate_resolution_rl(snrkey):
    keys = snr_groups.get(snrkey)
    if float(snrkey)<10:
        return None
    
    ones = np.ones(4)
    arr = []
    arr2 = []
    for key in keys:        
        arr.append(np.array([rl_bckg_sub(np.mean(im, axis=0)) for im in rldata[key]]))
        arr2.append(np.array([np.kron(np.mean(im, axis=0), ones) for im in inpdata[key]]))
    arr = np.mean(np.array(arr), axis=0)
    arr2 = np.mean(np.array(arr2), axis=0)
    arr /= np.max(arr)
    arr2 /= np.max(arr2)
    resolvable = np.array([is_somehow_resolvable_rl(y) for y in arr])
    inconclusive = False
    if resolvable[0]:
        inconclusive = True
    win = 1
    cres = [np.all(resolvable[i:i+win]) for i in range(len(resolvable)-win)]
    thr = np.where(cres)[0]
    if len(thr) > 0:
        thr = thr[0]
    else:
        thr = 0
    
    composite = np.zeros((41,200,3))
    composite[:,:,0] = arr**0.5
    composite[:,:,1] = arr2
    plt.imshow(composite)
    plt.title(snrkey)
    if not inconclusive:
        plt.hlines(thr, 0, 199, colors='r')
        plt.annotate(f'{distances[thr]/RAYLEIGH:.2f} R.d.', (20, thr-2), c='r')
        plt.show()
        plt.plot(arr[0]/np.max(arr[0]), c='k')
        plt.plot(arr[thr]/np.max(arr[thr]),"o-")
        plt.xlim(80,120)
        if thr > 1:
            plt.plot(arr[thr-1]/np.max(arr[thr-1]))
        plt.show()
    else:
        plt.title(snrkey+'???!')
        plt.show()
        plt.plot(arr[0])
        plt.show()
    plt.show()

for skey in sorted_snr_keys:
    plot_estimate_resolution_rl(skey)


# In[12]:


def estimate_resolution(snrkey):
    keys = snr_groups.get(snrkey)
    ones = np.ones(4)
    arr = []    
    for key in keys:
        estims = []
        for dframe in rldata[key]:            
            marginal = np.mean(dframe, axis=0)
            marginal /= np.max(marginal)
            resolvable = is_somehow_resolvable_rl(rl_bckg_sub(marginal))
            estims.append(resolvable)
        estims = np.array(estims)
        win = 2
        filtered_resolvability = [np.all(estims[i:i+win]) for i in range(len(estims)-win)]
        conclusive = (True in filtered_resolvability)
        if not conclusive:
            continue
        ires = np.where(filtered_resolvability)[0][0]
        if conclusive:
            arr.append(ires)
    succ_fraction = len(arr)/10
    return np.mean(distances[arr])/RAYLEIGH, succ_fraction

evaluated_performance_rl = []
for snrval, skey in zip(snr_table, sorted_snr_keys):
    x, s = estimate_resolution(skey)
    evaluated_performance_rl.append((snrval, x, s))
    print(f'{snrval:.1f} {x:.02f}, {s*100:.0f}%')

evaluated_performance_rl = np.array(evaluated_performance_rl)


# In[13]:


#manually done evaluation of thundershorm
# snr_1000_366    0.1
# snr_100_243 0.2
# snr_10_125  0.5
thstdata = np.array([
    [10,0.5],
    [100,0.2],
    [1000,0.1],
])


# In[14]:


plt.semilogx(evaluated_performance[:,0],evaluated_performance[:,1], "o-", label='cnn')
plt.semilogx(evaluated_performance_rl[1:,0], evaluated_performance_rl[1:,1], "s", label='rl')
plt.semilogx(thstdata[:,0], thstdata[:,1], "d", label='thst')
plt.legend()
plt.grid()
plt.xlabel('SNR')
plt.ylabel('$\delta_{R}$')
plt.xlim(3,1100)
plt.ylim(0,1)
plt.yticks(np.linspace(0,1,11))
plt.show()


# In[15]:


# plt.imshow(inpdata['noiseless'][-1])
classically_resolved_input = inpdata['noiseless'][-3][25]
classically_resolved_input /= np.max(classically_resolved_input)
plt.plot(classically_resolved_input)
plt.hlines([1-0.263], 0, 50)
plt.show()


# In[ ]:





# In[ ]:




