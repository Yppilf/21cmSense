import numpy as np
from matplotlib.pyplot import figure,colorbar
import matplotlib as mpl
from astropy import units as un
from astropy.units import Quantity
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum, hera
from py21cmsense.observatory import get_builtin_profiles, Observatory
import math

def generateSensitivityHera(freq, foreground="moderate", n_days=180, hpd=6):
    """
    Generate the sensitivity object for given observation

    Parameters:
    freq        - The frequency at which we observe in MHz
    foreground  - The approach to take for foreground excision. Default="moderate". Options = { "moderate", "optimistic" }
    n_days      - Amount of days observed. Default = 180
    hpd         - Hours per day of observation. Default = 6
    """
    sensitivity = PowerSpectrum(
        observation = Observation(
            observatory = Observatory(
                antpos = hera(hex_num=7, separation=(14*np.sqrt(3))/2*un.m, outriggers=True),
                beam = GaussianBeam(frequency=freq*un.MHz, dish_size=14*un.m),
                latitude=38*un.deg
            ),
            time_per_day = hpd*un.hour,
            n_days = n_days
        ),
        foreground_model = foreground,
    )

    return sensitivity

def generateSensitivityFromProfile(profile, freq, foreground="moderate", n_days=180, hpd=6):
    """
    Generate the sensitivity object for given observation

    Parameters:
    profile     - Profile of the observatory to use
    freq        - The frequency at which we observe in MHz
    foreground  - The approach to take for foreground excision. Default="moderate". Options = { "moderate", "optimistic" }
    n_days      - Amount of days observed. Default = 180
    hpd         - Hours per day of observation. Default = 6
    """
    obs = Observatory.from_profile(profile, frequency=75 * un.MHz)
    sensitivity = PowerSpectrum(
        observation = Observation(
            observatory = obs,
            time_per_day = hpd*un.hour,
            n_days = n_days
        ),
        foreground_model = foreground,
    )

    return sensitivity

def generateSensitivitySpectrum(sensitivity):
    power_std = sensitivity.calculate_sensitivity_1d()
    return power_std

def calc21ObsFreq(z):
    """Calculate the frequency at which we observe the 21cm signal for given redshift z"""
    vemit = 1420.4
    vobs = vemit/(1+z)
    return vobs

def plotScores(zList, scores, profileNames):
    """Plot the scores of each sensitivity simulation as a function of redshift for the different observatories. 

    Parameters:
    zList           - List of redshift values for which the scores are determined
    scores          - 2D list of all scores for all observatories
    profileNames    - List of all profile names

    Returns:
    None
    """
    transposedScores = np.array(scores).T
    fig = figure()
    frame = fig.add_subplot()
    for scoreList, name in zip(transposedScores, profileNames):
        frame.scatter(zList, scoreList, label=f"{name}")

    frame.set_xlabel("Redshift")
    frame.set_ylabel("Simulation score")
    frame.set_title("Score evolution for different observatories")
    frame.legend()

    fig.savefig("score_evolution.png")

def plotScores2(zList, scores, profileNames):
    """Plot the scores of each sensitivity simulation as a function of redshift for the different observatories. 

    Parameters:
    zList           - List of redshift values for which the scores are determined
    scores          - 2D list of all scores for all observatories
    profileNames    - List of all profile names

    Returns:
    None
    """
    transposedScores = np.array(scores).T
    fig = figure(figsize=(15,10))
    frame = fig.subplots(1, len(profileNames), sharex=True)
    i = 0
    for scoreList, name in zip(transposedScores, profileNames):
        frame[i].scatter(zList[1:], scoreList[1:])  # We remove the first element since they are consistently an outlier
        frame[i].set_title(f"{name}")
        frame[i].get_yaxis().get_major_formatter().set_useOffset(False)
        i+=1


    fig.supxlabel("Redshift")
    fig.supylabel("Simulation score")
    fig._supxlabel.set_size(25)
    fig._supylabel.set_size(25)
    fig.suptitle("Scores of simulations as a function of redshift")
    fig._suptitle.set_size(20)      # change figure title size

    fig.tight_layout()

    fig.savefig("score_evolution.png")

def filterInf(dataset):
    newX = []
    newY = []
    xList = dataset[0]
    yList = dataset[1]

    for x,y in zip(xList, yList):
        if not math.isinf(y.value) and not y.value == 0 and not x.value == 0:
            newX.append(x)
            newY.append(y)
    return (newX, newY)

def filterZero(dataset):
    newX = []
    newY = []
    xList = dataset[0]
    yList = dataset[1]

    for x,y in zip(xList, yList):
        if not y == 0 and not x == 0:
            newX.append(x)
            newY.append(y)
    return (newX, newY)

def is_nan(value):
    try:
        return math.isnan(value)
    except TypeError:
        return np.isnan(value)

def normalize(arr):
    # Replace nan with 0 while maintaining units
    cleaned_arr = []
    for i in arr:
        if isinstance(i, Quantity):
            if is_nan(i.value):
                cleaned_arr.append(0 * i.unit)  # Replace nan with 0 and keep the unit
            else:
                cleaned_arr.append(i)
        else:
            cleaned_arr.append(0 if is_nan(i) else i)
    
    # Convert astropy quantities to their numerical values
    arr_values = [i.value if isinstance(i, Quantity) else i for i in cleaned_arr]
    arr_values = np.array(arr_values)
    
    # Normalize the array
    min_val = np.min(arr_values)
    max_val = np.max(arr_values)
    
    if max_val - min_val == 0:
        # Avoid division by zero if all values are the same
        normalized_arr = np.zeros_like(arr_values)
    else:
        normalized_arr = (arr_values - min_val) / (max_val - min_val)
    
    return normalized_arr

def scoreUncertainties(datasetList):
    """Assigns a score to each dataset for ranking.
    The ranking takes into account the mean and median y values, the standard deviation in y,
    the range in y values and the minimum x value.
    y represents the simulated uncertainty and x the wave number.

    Parameters:
    datasetList     - The list of tuples containing x and y for the different observations

    Returns:
    The scores in the same order as the sensitivityList
    """
    # Format dataset
    # There is an issue that normalizing always creates a zero. The zero will be filtered out.
    # Since it gets filtered out for every dataset, all datasets will be affected approximately equally
    normalized_datasets = [filterZero((normalize(x), normalize(y))) for x, y in datasetList]

    
    avg_y_values = [np.mean(y) for x, y in normalized_datasets]
    median_y_values = [np.median(y) for x, y in normalized_datasets]
    std_y_values = [np.std(y) for x, y in normalized_datasets]
    ranges = [np.max(y) - np.min(y) for x, y in normalized_datasets]
    min_x_values = [np.min(x) for x, y in normalized_datasets]
    len_x_values = [len(x) for x,y in normalized_datasets]
    
    max_avg_y = np.max(avg_y_values)
    max_median_y = np.max(median_y_values)
    max_std_y = np.max(std_y_values)
    max_range = np.max(ranges)
    min_x = np.min(min_x_values)
    max_len_x = np.max(len_x_values)
    
    scores = []
    for i, (x, y) in enumerate(normalized_datasets):
        Y_mean_score = 1 - avg_y_values[i] / max_avg_y
        Y_median_score = 1 - median_y_values[i] / max_median_y
        Y_score = (Y_mean_score + Y_median_score) / 2
        
        SD_score = 1 - std_y_values[i] / max_std_y
        R_score = 1 - ranges[i] / max_range
        X_score = min_x / min_x_values[i]
        XR_score = len(x)/max_len_x
        
        final_score = 0.4 * Y_score + 0.2 * SD_score + 0.1 * R_score + 0.2 * X_score + 0.1 * XR_score
        scores.append(final_score)
    
    return scores

def observatorySubplot(frame, observatory, obsName):
    frame.scatter(observatory.baselines_metres[:,:, 0], observatory.baselines_metres[:,:,1], alpha=0.1)
    frame.set_title(f"{obsName}")
    return frame

def baselineSubplot(frame, observatory):
    red_bl = observatory.get_redundant_baselines()
    baseline_group_coords = observatory.baseline_coords_from_groups(red_bl)
    baseline_group_counts = observatory.baseline_weights_from_groups(red_bl)

    sc = frame.scatter(baseline_group_coords[:,0], baseline_group_coords[:,1], c=baseline_group_counts)
    return sc

def observatoryMosaic(sensitivityList, profileNames):
    fig = figure(figsize=(25,10))
    frame = fig.subplots(2, len(profileNames))

    for i in range(len(profileNames)):
        observatory = sensitivityList[i].observation.observatory
        for j in range(2):
            if j == 1:
                sc = baselineSubplot(frame[j,i], observatory)
            else:
                observatorySubplot(frame[j,i], observatory, profileNames[i])

    fig.suptitle("Maps of all observatories used")
    fig._suptitle.set_size(20)

    fig.supxlabel("Baseline Length [x, m]")
    fig.supylabel("Baseline Length [y, m]", x=0.08)
    fig._supxlabel.set_size(20)
    fig._supylabel.set_size(20)

    # Add colorbar to plot
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.1, 0.03, 0.8])
    cbarPlot = fig.colorbar(sc, cax=cbar_ax)
    cbarPlot.ax.tick_params(labelsize=15)
    cbarPlot.set_label('Number of baselines in group',fontsize=20)

    fig.savefig(f"observatories_mosaic.png")

def mosaicPlot(minZ, maxZ, nZ, n_days, hpd):
    """See how the uncertainty changes at various values for k for different redshifts

    Parameters:
    minZ    - Minimum redshift for evaluation
    maxZ    - Maximum redshift for evaluation
    nZ      - Amount of different redshifts to plot
    n_days  - Number of days observed
    hpd     - Hours per day observed
    

    Returns:
    None
    """
    zList = np.linspace(minZ, maxZ, nZ)
    profileNames = get_builtin_profiles()
    profileNames.append("HERA")

    f = open("datasets.txt", "w")

    allScores = []
    for z in zList:
        freq = calc21ObsFreq(z)
        sensitivities = [generateSensitivityFromProfile(profile, freq, n_days=n_days, hpd=hpd) for profile in get_builtin_profiles()]
        sensitivity_hera = generateSensitivityHera(freq=freq, n_days=n_days, hpd=hpd) 
        sensitivities.append(sensitivity_hera)
        datasets = [filterInf((sensitivity.k1d, generateSensitivitySpectrum(sensitivity))) for sensitivity in sensitivities]
        scores = scoreUncertainties(datasets)
        allScores.append(scores)
        for item in datasets:
            # Convert the tuple to a string and write it to the file
            values = ([q.value for q in item[0]], [q.value for q in item[1]])
            f.write(f"{values}\n")
        
    f.close()
    np.savetxt("scores.txt", allScores)

    plotScores(zList, allScores, profileNames)
    observatoryMosaic(sensitivities, profileNames)

def mosaicFromTxt(minZ, maxZ, nZ):
    """Load mosaic from txt
    Note that the redshifts used must correspond to the dataset, there are no checks
    """
    allScores = np.loadtxt("scores.txt")
    zList = np.linspace(minZ, maxZ, nZ)
    profileNames = get_builtin_profiles()
    profileNames.append("HERA")
    plotScores2(zList, allScores, profileNames)

mosaicFromTxt(10,22,24)
# mosaicPlot(10, 22, 24, 540, 6)
