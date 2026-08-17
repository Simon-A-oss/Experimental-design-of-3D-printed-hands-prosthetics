""""
============================================================================================================
READMEEEEEE
Ce fichier est actuellement la dernière version de ce code
Ce code a pour but d'analyser les données de forces enregistrées dans un fichier TDMS.
Il permet de:
    - inspecter la structure du fichier et les statistiques de chaque canal
    - détecter les "time jumps" dans les données de timestamp
    - visualiser les forces sur quelques périodes centrées sur leurs pics
    - visualiser le signal de chaque force sur base uniquement de ses pics
    - in progress: analyser les variations moyennes des pics de force avec une moyenne mobile

TODO:
    - regarder pour ref force et interesting force dans les variables et la main, je suis pas sur de l'intéret des deux
    - faire la détection des changements brutaux de pics (via la moyenne mobile)
    - peut etre faire une fonction qui utilise un peu tout: 
        - display l'evolution des pics au cours du temps pour toutes les forces
        - détecter les changements brutaux de pics sur chaque force
        - afficher les périodes de la force qui a eu un pic avant et apres les changements brutaux, pour regarder si la forme du signal change aussi
============================================================================================================
"""
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from nptdms import TdmsFile
from scipy.signal import find_peaks, butter, filtfilt 
from scipy.ndimage import uniform_filter1d




# ===== variables =====############################################################################################################
filepath = "Runs\\FatigueTests\\RunBaseVsV1\\TEST0_2025_12_22_15_3_23.tdms"
filepath = "Runs\\FatigueTests\\RunBaseModel-Ultimaker\\TEST0_2025_12_22_17_31_50.tdms"
filepath = "Runs\\FatigueTests\\Runbleus\\TEST0_2025_12_22_17_22_21.tdms"


forces_to_show = [1,2,3,4,5,6]                # forces to show in the plots (1 to 6)
framerate = 100
finger_frequency = 2

# file inspection
show_structure = False              # basic display of the structure of the TDMS file and basic statistics of each channel
test_time_jumps = False              # test for time jumps in the timestamp data, which can cause problems for the analysis of the forces, because it can create artificial peaks in the force signals that are not actually present in the real signal, and can also create gaps in the data that can affect the detection of peaks and the calculation of their values

# functions to use
show_full_forces = False            # shows the full signals over the entire test (this will lag)
show_periods = False                 # show the force signals on a few periods centered on their peaks, to better visualize the shape of the signal at various moments of the recording
show_peaks = True                  # show the evolution over time of the force measured
show_mean_peaks = False             # debug to test the event detection in the video recording


# parameters for period analysis functions
ref_force = 1                       # reference force channel to analyse
Time_start = 100*20                 # start time INDEX
Time_step = 600*99                  # step between each plot (in points)
steps = 6                           # number of plots to show
interesting_times = [40000, 40720]  # 

# parameters for peaks and mean peaks analysis functions
peaks_start = 0                     # start index for the analysis of peaks and mean peaks, set to 0 to analyze the entire signal
peaks_end = -1                      # set to -1 to analyze the entire signal, or set to a specific index to analyze a specific portion




# ===== functions =====#############################################################################################################

def detect_time_jumps(signal, threshold=50):
    """
    Description:
    Detect time jumps in the given signal based on a specified threshold.
    
    parameters:
        signal (array-like): input signal to analyze
        threshold (float): threshold value to identify jumps
    
    return:
        jumps (list): list of indices where time jumps are detected
        sizes (list): list of jump sizes at the detected indices
    """
    jumps = []
    sizes = []
    for i in range(1, len(signal)):
        if abs(signal[i] - signal[i-1]) > threshold:
            jumps.append(i)
            sizes.append(signal[i] - signal[i-1])
    if jumps:
        print("Time jumps detected at indices:", jumps)
        print("Jump sizes:", sizes)
    else:
        print("No time jumps detected.")
    return jumps, sizes

def show_allForces_fullPeriods(ref_force=1, Time_start=18000, Time_step=600*99, steps=6):
    """
    Description:
    plot all forces over a bit more than one period, centered on the reference force channel peaks
    parameters:
        ref_force (int): reference force channel to analyse (1 to 6)
        Time_start (int): start time INDEX 
        Time_step (int): step between each plot (in points)
        steps (int): number of plots to show
    
    return:
        nothing, this function generates plots

    notes:
        so far so good :)
    """

    Time_frame = 55         # a period is about 50ms, so 55 points to have a bit more than one period
    Times = tdms["myRIO_Data"]["Timestamp (s)"] - tdms["myRIO_Data"]["Timestamp (s)"][Time_start]  # Normaliser le temps pour commencer à 0
    filtered_ref_force = lowpass(forces[ref_force - 1], fs=1000, fc=10) #- 1.24)*20

    fig, axes = plt.subplots(nrows=steps, ncols=1, figsize=(10, 2.5*steps), sharex=False)

    for k in range(steps):
        sample_start = Time_start + k * Time_step
        sample_end = sample_start + Time_frame -1

        filtered_force = filtered_ref_force[sample_start:sample_end]
        peaks, properties = find_peaks(filtered_force, height=0, distance=spacing) 

        center = (peaks[0]+sample_start) if len(peaks) > 0 else 0
        start = center - Time_frame//2
        end = center + Time_frame//2
        # adjusted_filtered_force = filtered_ref_force[start:end]

            
        time = Times[start:end]
        ax = axes[k]

        #ax.plot(time, adjusted_filtered_force, label=f"filtered Force 1")
        for i in range(0, 6):
            force = forces[i][start:end]
            ax.plot(time, force, label=f"Force {i+1}")
        ax.set_title(f"Fenêtre {k+1} (index {sample_start})")
        ax.set_ylabel("Force [N]")
        ax.grid(True)

        if k == 0:
            ax.legend(loc="upper right", fontsize=8)


    ax.set_xlabel("Temps [s]")
    plt.tight_layout()
    plt.show()


def show_force_fullPeriods(force_id=1, forces=None, number_of_peaks=500):
    """
    Description:
    parameters:
        /
    return:
        nothing, this function prints peak info
    """
    max_index = len(tdms["myRIO_Data"]["Timestamp (s)"])
    filtered_force = lowpass(forces[force_id - 1], fs=1000, fc=10)
    peaks, properties = find_peaks(filtered_force, height=0, distance=spacing)

    selected_peaks = peaks[::len(peaks)//number_of_peaks][:number_of_peaks]  # x pics répartis
    Time_frame = 55         # a period is about 500ms (closer to 520ms), so at 10ms sampling frequency, take 55 points to have a bit more than one period
    half_frame = Time_frame//2

    plt.figure(figsize=(8, 4))

    for p in selected_peaks:
        start = p - half_frame
        end = p + half_frame

        if p-half_frame < 0 or p+half_frame> max_index:
            continue

        signal = forces[force_id - 1][p-half_frame:p+half_frame]
        t = np.linspace(-0.5, 0.5, len(signal))
        plt.plot(t, signal, alpha=0.5, label=f"periode #{p}")            

    plt.axvline(0, color='k', linestyle='--', label='Pic')
    plt.legend(loc="upper right", fontsize=8)
    plt.xlabel("Temps relatif [s]")
    plt.ylabel("Force [N]")
    plt.title(f"Superposition de {number_of_peaks} périodes alignées sur le pic")
    plt.grid(True)
    plt.show()

def show_force_fullPeriods2(force_id=1, forces=None, timestamps=None, number_of_peaks=500):
    """
    Description:
    parameters:
        timestamps : array-like, same length as forces[i], giving the
                      timestamp (in seconds) of each force sample.
    return:
        nothing, this function prints peak info
    """
    max_index = len(tdms["myRIO_Data"]["Timestamp (s)"])
    filtered_force = lowpass(forces[force_id - 1], fs=1000, fc=10)
    peaks, properties = find_peaks(filtered_force, height=0, distance=spacing)

    selected_peaks = peaks[::len(peaks)//number_of_peaks][:number_of_peaks]
    Time_frame = 55
    half_frame = Time_frame // 2

    fig, ax = plt.subplots(figsize=(8, 4))

    n = len(selected_peaks)
    cmap = cm.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, n))

    # Real timestamp (in seconds) of each selected peak, used to label the
    # colorbar. Falls back to peak index if no timestamps array is given.
    if timestamps is not None:
        peak_times = np.asarray(timestamps)[selected_peaks]
    else:
        peak_times = selected_peaks.astype(float)

    for i, p in enumerate(selected_peaks):
        start = p - half_frame
        end = p + half_frame

        if start < 0 or end > max_index:
            continue

        signal = forces[force_id - 1][start:end]
        t = np.linspace(-0.5, 0.5, len(signal))
        ax.plot(t, signal, alpha=0.5, color=colors[i])

    ax.axvline(0, color='k', linestyle='--', label='Period Center')

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=peak_times.min(), vmax=peak_times.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Peak time [s]" if timestamps is not None else "Peak index")

    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Relative time [s]")
    ax.set_ylabel("Force [N]")
    ax.set_title(f"Superposition of {number_of_peaks} periods aligned on peak")
    ax.grid(True)
    plt.show()

def show_force_specific_periods(force_id=1, interest_times=None):
    """
    Description:
        show the force signal over specific periods closer to the specified interest times, to better visualize the shape of the signal around particular changes in peak values
    parameters:
        force_id (int): ID of the force channel to analyze (1 to 6)
        interest_times (array-like): times of the detected peaks to focus on
    """
    Time_frame = 55         # a period is about 500ms (closer to 520ms), so at 10ms sampling frequency, take 55 points to have a bit more than one period
    half_frame = Time_frame//2
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Couleurs pour les différentes périodes
    plt.figure(figsize=(8, 4))

    for t in interest_times:
        color = random.choice(colors)
        #find the index of the specified time
        index = t*100 # not completely accurate because the sampling somehow does not take 1000 points per second, but it's close enough for this purpose
        search_timeframe_start = index-520 
        search_timeframe_end = index+520 # this gives a search timeframe of about 10 seconds
        peaks, properties = find_peaks(forces[force_id - 1][search_timeframe_start:search_timeframe_end], height=0, distance=spacing)
        if len(peaks) == 0:
            print(f"No peaks found around time {t}")
            continue
        
        #plotting the periods of all the peaks found in the search timeframe, to avoid plotting a period that is not representative of the actual trend of the signal around the specified time
        for p in peaks:
            peak = search_timeframe_start + p
            start = peak - half_frame
            end = peak + half_frame
            if start < 0 or end > len(forces[force_id - 1]):
                print(f"Peak at time {t} is too close to the start or end of the signal, skipping.")
                continue

            signal = forces[force_id - 1][start:end]
            time = np.linspace(-0.5, 0.5, len(signal))
            plt.plot(time, signal, color=color, alpha=0.5, label=f"Pic #{peak}")            


    
    plt.axvline(0, color='k', linestyle='--', label='Pic')
    plt.legend(loc="upper right", fontsize=8)
    plt.xlabel("Temps relatif [s]")
    plt.ylabel("Force [N]")
    plt.title(f"Superposition de périodes spécifiques de la force {force_id} alignées sur leur pic ")
    plt.grid(True)
    plt.show()

def show_multiple_forces_peaks(list_of_forces=[1,2,3,4,5,6]):
    """
    Description:
    parameters:
        /
    return:
        nothing, this function prints peak info

    notes:
        so far so good :)
    """

    # plot l'evolution des forces sur base uniquement des pics
    plt.figure(figsize=(10, 5))
    for j in list_of_forces:
        i = j-1
        peaks, properties = find_peaks(forces[i], height=0, distance=spacing)
        peak_times = times[peaks]
        peak_values = forces[i][peaks]

        plt.plot(peak_times, peak_values, marker='.', markersize=0.4, linestyle='-', linewidth=0.1,alpha = 0.2, label=f"Force {i+1} Peaks")
        
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid(True)
    plt.legend()
    plt.title(f"Forces Peaks Over Time")
    plt.tight_layout()
    plt.show()

def show_force_peaks(force_id=1):
    """
    Description:
        Show the peaks of a specific force signal over time.
    parameters:
        force_id (int): ID of the force channel to analyze (1 to 6)
    return:
        peak_times (array-like): times of the detected peaks
        peak_values (array-like): values of the detected peaks 
    """

    # plot l'evolution des forces sur base uniquement des pics
    real_peaks, real_properties = find_peaks(forces[force_id - 1], height=0, distance=spacing)
    real_peak_times = times[real_peaks]
    real_peaks_values = forces[force_id - 1][real_peaks]

    
    plt.figure(figsize=(10, 5))
    # show raw signal
    #plt.plot(times, forces[force_id - 1],'.', markersize=0.5,  linestyle='-', linewidth=0.1, label=f"Force {force_id} Raw Signal")
    plt.plot(real_peak_times, real_peaks_values, marker='.', markersize=1, linestyle='-', linewidth=0.1, label=f"Force {force_id} Real Peaks", alpha = 0.5) 
    plt.title(f"Force {force_id} Peaks Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()
    #plt.ylim(0,7.5)
    plt.show()
    return real_peak_times, real_peaks_values    

def show_peaks_moving_average(signal,force_id=1, big_window_size=2000, small_window_size=100 , threshold_enable=0.25, threshold_disable=0.10):
    """
    Description:
        show the moving average of the peaks of the signal on top of the original peaks, to better visualize peaks as a in-between function to detect sharp changes in force

    parameters:
        signal (array-like) : input signal to analyze [1:6]
        big_window_size (int): size of the big moving average window
        small_window_size (int): size of the small moving average window
    

    return:
        nothing, this function generates a plot
    """    

    """peak_indexes, properties = find_peaks(signal, height=0, distance=40)
    peak_values = signal[peak_indexes] 
    peak_times = times[peak_indexes]"""
    peak_values = signal
    peak_times = times

    small_averages = []
    big_averages = []
    deltas = []
    poi = []

    Zoi = False
    for i in range(0, len(peak_values)):
        small_average = np.mean(peak_values[max(0, i-small_window_size):i])
        big_average = np.mean(peak_values[max(0, i-big_window_size):i])
        delta = abs(big_average - small_average)
        if delta > threshold_enable:
            if not Zoi:
                Zoi = True
                poi.append(i)
        else:
            if Zoi and poi[-1] < i-600 and delta < threshold_disable: 
                # for each poi, we want to keep 5 mins of video before and after it, so if two poi are less than 5 mins appart... 
                Zoi = False
        small_averages.append(small_average)
        big_averages.append(big_average)
        deltas.append(delta)

    #print(f"longueur de peaks: {len(peaks)}, longueur de peak_values: {len(peak_values)}, longueur de small_averages: {len(small_averages)}, longueur de big_averages: {len(big_averages)}, longueur de deltas: {len(deltas)}")

    #this block takes the average AROUND the peak, not just before it. this is arguably better and faster for sharp changes detection,
    # but doesn't mimmic the way it is calculated in the MyRIO, so i dont rly intend to use it for now
    """big_average = uniform_filter1d(peak_values, size=big_window_size)
    small_average = uniform_filter1d(peak_values, size=small_window_size)
    delta = abs(big_average - small_average) - 0.5

    poi = np.where((abs(big_average - small_average) - 0.5) > 0)[0]
    print(f"Points of interest (potential sharp changes in force): {poi}, corresponding to times: {peak_times[poi]} and peak values: {peak_values[poi]}")#"""

    plt.figure(figsize=(10, 5))
    for i in range(len(peak_times)):
        if i in poi:
            plt.axvline(peak_times[i], color='r', linestyle='--', linewidth=1, label='Potential Sharp Change' if i == poi[0] else "")
            for j in range(1200):
                plt.axvline(peak_times[min(max(i-j+600, 0), len(peak_times)-1)], color='0.8', linestyle='-', linewidth=1, alpha=0.8, label='Potential Sharp Change' if i+j == poi[0] else "" ) 
    
    plt.plot(peak_times, peak_values, marker='.', markersize=0.5, linestyle='-', linewidth=0.8, label='Original Peaks')
    plt.plot(peak_times, small_averages, marker='.', markersize=0.5, linestyle='-', linewidth=0.5, label='small moving Average')
    plt.plot(peak_times, big_averages, marker='.', markersize=0.5, linestyle='-', linewidth=0.5, label='big moving Average',)
    plt.plot(peak_times, deltas, color = 'y',marker='.', markersize=1, linestyle='-', linewidth=0.1, label='Delta')
    plt.title(f"Peaks and Moving Averages")
    plt.xlabel('Time [s]')
    plt.ylabel('Peak Value')
    plt.legend(loc = "upper right", fontsize=8)
    plt.grid(True)
    plt.ylim(0,7.5)
    plt.show()

def lowpass(signal, fs, fc=10):
    """
    Description:
        applies a lowpass filter to the signal
        no idea of how it works exactly, this was made by chatgpt, 

    parameters:
        signal (array-like) : input signal to filter
        fs (float): sampling frequency of the signal
        fc (float): cutoff frequency of the lowpass filter
    
    return:
        (array-like): filtered version of the given signal
    
    notes: 
    fc could be the cutoff frequency, fs sampling frequency, but the calculation in the butter function is weird"""

    b, a = butter(2, fc/(fs/2), btype='low')
    return filtfilt(b, a, signal) 


# ===== main code =====#############################################################################################################

print(f"reading file {filepath}")
tdms = TdmsFile.read(filepath)

spacing = framerate/finger_frequency

# Affichage de la structure du fichier TDMS et des statistiques de chaque canal
if show_structure:
    for group in tdms.groups():
        print("Groupe :", group.name)
        for channel in group.channels():
            print("  Channel :", channel.name)
            print("    Nombre de points :", len(channel))
            print("    Valeur moyenne :", channel[:].mean())

# Test for time jumps in the timestamp data
if test_time_jumps:
    timestamps = tdms["myRIO_Data"]["Timestamp (s)"]
    jumps, sizes = detect_time_jumps(timestamps, 5)

# extract forces and times for analysis
if show_periods or show_peaks or show_mean_peaks or True:
    print(f"extracting data for signals {forces_to_show}")
    forces = [tdms["myRIO_Data"][f"Force {i} (N)"] for i in range(1, 7)]
    times = tdms["myRIO_Data"]["Timestamp (s)"]- tdms["myRIO_Data"]["Timestamp (s)"][0]
a = peaks_start
b = peaks_end if peaks_end != -1 else len(times)
forces = [force[a:b] for force in forces]
times = times[a:b]    
    
if show_periods:
    for interesting_force in forces_to_show:
        #show_force_specific_periods(interesting_force, interesting_times)
        #for force_id in range(1, 7):
            #show_force_fullPeriods(force_id, forces)
        show_force_fullPeriods2(4, forces,times)
        #show_allForces_fullPeriods(ref_force, Time_start, Time_step, steps)
    
if show_peaks:
    """for force_id in forces_to_show:
        show_force_peaks(force_id)"""
    show_multiple_forces_peaks(forces_to_show)

if show_mean_peaks:
    for i in range(0, 6):
        show_peaks_moving_average(forces[i][a:b], i+1, 200000, 10000,  0.08, 0.25)


if show_full_forces:

    plt.figure(figsize=(10, 5))
    for i in forces_to_show:
        plt.plot(times, forces[i-1], label=f"Force {i}", alpha = 0.5)

    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.title("All Forces Over Time")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
