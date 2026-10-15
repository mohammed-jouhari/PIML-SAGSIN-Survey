"""Physics of an underwater acoustic link (Thorp absorption, Stojanovic attenuation and noise).

Units: distance l in km, frequency f in kHz, losses in dB.
References: M. Stojanovic, "On the relationship between capacity and distance in an underwater
acoustic communication channel," ACM SIGMOBILE MC2R, vol. 11, no. 4, pp. 34-43, 2007.
"""
import numpy as np


def thorp_db_per_km(f_khz):
    """Thorp absorption coefficient 10 log10 a(f) in dB/km (f in kHz)."""
    f2 = np.asarray(f_khz, dtype=float) ** 2
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


def transmission_loss_db(l_km, f_khz, k=1.5, gamma=1.0, c0=0.0, l_ref_km=1e-3):
    """10 log10 A(l, f) = 10 k log10(l / l_ref) + gamma * l * [10 log10 a(f)] + c0.

    k is the spreading factor (1 cylindrical, 2 spherical, 1.5 practical), gamma scales the
    Thorp absorption, c0 is a constant offset. With gamma = 1 and c0 = 0 this is the
    Stojanovic attenuation model with distance referred to 1 m.
    """
    l = np.asarray(l_km, dtype=float)
    return 10 * k * np.log10(l / l_ref_km) + gamma * l * thorp_db_per_km(f_khz) + c0


def ambient_noise_db(f_khz, shipping=0.5, wind_mps=0.0):
    """Ambient noise power spectral density in dB re 1 uPa^2/Hz (turbulence, shipping, waves, thermal)."""
    f = np.asarray(f_khz, dtype=float)
    nt = 17 - 30 * np.log10(f)
    ns = 40 + 20 * (shipping - 0.5) + 26 * np.log10(f) - 60 * np.log10(f + 0.03)
    nw = 50 + 7.5 * np.sqrt(wind_mps) + 20 * np.log10(f) - 40 * np.log10(f + 0.4)
    nth = -15 + 20 * np.log10(f)
    return 10 * np.log10(sum(10 ** (x / 10) for x in (nt, ns, nw, nth)))
