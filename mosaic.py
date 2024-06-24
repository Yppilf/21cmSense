import numpy as np
from matplotlib.pyplot import figure,colorbar
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from astropy import units as un
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum, hera

# Create a mosaic showing how different variables affect the uncertainty generated by 21cmSENSE for HERA
def calc21ObsFreq(z):
    """Calculate the frequency at which we observe the 21cm signal for given redshift z"""
    vemit = 1420.4
    vobs = vemit/(1+z)
    return vobs

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

def generateSensitivitySpectrum(sensitivity):
    power_std = sensitivity.calculate_sensitivity_1d()
    return power_std

def subplot(frame, n_days, min_hpd, max_hpd, n_hpd, z, i, j, n, cmap):
    hpdList = np.linspace(min_hpd, max_hpd, n_hpd)
    
    freq = calc21ObsFreq(z)
    foregrounds = ["moderate", "optimistic"]
    markers = ["^", "s"]
    for k, hpd in enumerate(hpdList):
        for l, fg in enumerate(foregrounds):
            sensitivity_hera = generateSensitivityHera(freq=freq, foreground=fg, n_days=n_days, hpd=hpd) 
            std_spectrum_1d = generateSensitivitySpectrum(sensitivity_hera)
            kList = sensitivity_hera.k1d
            frame.scatter(kList, std_spectrum_1d, s=5, label=f"{hpd:.1f}", color=cmap(k/n_hpd), marker=markers[l])

    if i == 0:
        frame.set_title(f"{n_days:.0f} days")
        frame.title.set_size(15)
    if j == 0:
        frame.set_ylabel(f"z={z:.2f}")
        frame.yaxis.label.set_size(15)

    frame.set_yscale('log')
    frame.set_xscale('log')

    if j == 0 and i == n:
        frame.tick_params(top=False, bottom=True, left=True, right=False, labelsize=15)
    elif j == 0:
        frame.tick_params(top=False, bottom=False, left=True, right=False, labelsize=15)
    elif i == n:
        frame.tick_params(top=False, bottom=True, left=False, right=False, labelsize=15)
    else:
        frame.tick_params(top=False, bottom=False, left=False, right=False, labelsize=15)

    return frame

def mosaicPlot(minZ, maxZ, minDays, maxDays, min_hpd, max_hpd, n_hpd, nPlots):
    """See how the uncertainty changes at various values for k for different redshifts

    Parameters:
    minZ    - Minimum redshift for evaluation
    maxZ    - Maximum redshift for evaluation
    minDays - Minimum number of days observed
    maxDays - Maximum number of days observed
    min_hpd - Minimum hours per day observed
    max_hpd - Maximum hours per day observed
    n_hpd   - Amount of different hours per day
    nPlots  - Amount of subplots to make per row

    Returns:
    None
    """
    # Calculate a proper uniform colormap
    cmap = mpl.cm.cool_r   # Create a colormap
    norm = mpl.colors.Normalize(vmin=min_hpd, vmax=max_hpd) # Normalize your values to fit the colormap
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy empty array to force the colorbar to use the given colormap
    
    zList = np.linspace(minZ, maxZ, nPlots)
    daysList = np.linspace(minDays, maxDays, nPlots)

    fig = figure(figsize=(10,10))
    frame = fig.subplots(nPlots, nPlots, sharex=True, sharey=True, gridspec_kw = {'wspace':0, 'hspace':0})

    for i,z in enumerate(zList):
        for j,n_days in enumerate(daysList):
            subplot(frame[i,j], n_days, min_hpd, max_hpd, n_hpd, z, i, j, nPlots-1, cmap)

    fig.supxlabel("k [h/Mpc]")
    fig.supylabel(r'$\delta \Delta^2_{21}$')
    fig._supxlabel.set_size(25)
    fig._supylabel.set_size(25)

    fig.suptitle("Analysis of expected total uncertainty for HERA observatory")
    fig._suptitle.set_size(20)      # change figure title size

    fig.tight_layout()

    # Add colorbar to plot
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbarPlot = fig.colorbar(sm, cax=cbar_ax)
    cbarPlot.ax.tick_params(labelsize=15)
    cbarPlot.set_label('Hours per day',fontsize=20)

    fig.savefig(f"HERA_observatory_analysis.png")
        
mosaicPlot(10, 22, 100, 800, 6, 24, 6, 5)

