import numpy as np
from matplotlib.pyplot import figure,colorbar
from astropy import units as un
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum, hera

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
    power_std_thermal = sensitivity.calculate_sensitivity_1d(thermal=True, sample=False)
    power_std_sample = sensitivity.calculate_sensitivity_1d(thermal=False, sample=True)
    return power_std, power_std_thermal, power_std_sample

def documentation():
    print(help(PowerSpectrum))
    # print(help(Observation))
    # print(help(Observatory))
    # print(help(hera))

def calc21ObsFreq(z):
    """Calculate the frequency at which we observe the 21cm signal for given redshift z"""
    vemit = 1420.4
    vobs = vemit/(1+z)
    return vobs

def displayObservatory(observatory, observatoryName):
    red_bl = observatory.get_redundant_baselines()
    baseline_group_coords = observatory.baseline_coords_from_groups(red_bl)
    baseline_group_counts = observatory.baseline_weights_from_groups(red_bl)

    fig = figure()
    frame = fig.add_subplot()
    sc = frame.scatter(baseline_group_coords[:,0], baseline_group_coords[:,1], c=baseline_group_counts)
    cbar = fig.colorbar(sc)
    cbar.set_label("Number of baselines in group", fontsize=15)
    frame.set_title(f"Map of {observatoryName}")
    fig.tight_layout()
    fig.savefig(f"Observatory_redundancy_{observatoryName}.png")


    fig = figure()
    frame = fig.add_subplot()
    frame.scatter(observatory.baselines_metres[:,:, 0], observatory.baselines_metres[:,:,1], alpha=0.1)
    frame.set_xlabel("Baseline Length [x, m]")
    frame.set_ylabel("Baseline Length [y, m]")
    frame.set_title(f"Observatory map {observatoryName}")
    fig.tight_layout()
    fig.savefig(f"Observatory_map_{observatoryName}.png")

def plotSpectrum(sensitivity, observatoryName, z, n_days):
    std_spectrum_1d, std_spectrum_thermal, std_spectrum_sample = generateSensitivitySpectrum(sensitivity)
    fig = figure()
    frame = fig.add_subplot()

    frame.scatter(sensitivity.k1d, std_spectrum_1d, label='Total')
    frame.scatter(sensitivity.k1d, std_spectrum_thermal, label='Thermal')
    frame.scatter(sensitivity.k1d, std_spectrum_sample, label='Sample')

    frame.set_title(f"Sensitivities for the 21cm spectrum at z={z}\nObservation time = {n_days} days")
    frame.set_xlabel("k [h/Mpc]")
    frame.set_ylabel(r'$\delta \Delta^2_{21}$')
    frame.set_yscale('log')
    frame.set_xscale('log')
    # plt.xlim(0,1)
    frame.legend()
    fig.tight_layout()
    fig.savefig(f"sensitivitySpectrum_{observatoryName}.png")

def main():
    z = 16.1
    n_days = 540
    freq = calc21ObsFreq(z)
    sensitivity_hera = generateSensitivityHera(freq=freq, n_days=n_days) 
    displayObservatory(sensitivity_hera.observation.observatory, "HERA")
    plotSpectrum(sensitivity_hera, "HERA", z, n_days)

main()
# documentation()