####################################################################################################
#                                      _seq_fixtures.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-17                                                                              #
#                                                                                                  #
# Purpose: Builds a tiny, real Pulseq ".seq" file for the tests in this directory - not a test      #
#          module itself (no "test_" prefix), so pytest does not collect it.                        #
#                                                                                                  #
####################################################################################################

import numpy as np


def write_synthetic_seq(path: str, n_samples: int = 64, fov_m: float = 0.01,
                        b0_tesla: float = 3.0, n_shots: int = 1) -> None:
    """
    Write a minimal real ".seq" file: "n_shots" repeats of one slice-selective
    excitation followed by one readout gradient with an ADC window - enough
    for pypulseq to report a nonzero gradient waveform and a k-space
    trajectory, without being a clinically meaningful sequence.

    Args:
        path: Where to write the ".seq" file.
        n_samples: ADC samples per readout.
        fov_m: Declared field of view (isotropic), meters.
        b0_tesla: Declared "B0" definition, Tesla.
        n_shots: Number of excitation+readout repeats.
    """
    import pypulseq as pp

    system = pp.Opts(max_grad=30, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
                     rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)
    seq = pp.Sequence(system=system)

    for shot in range(n_shots):
        rf, gz, _ = pp.make_sinc_pulse(
            flip_angle=np.pi / 2, duration=1e-3, slice_thickness=fov_m,
            apodization=0.5, time_bw_product=4, system=system, return_gz=True)
        gx = pp.make_trapezoid(channel='x', flat_area=(1.0 + 0.1 * shot) / fov_m,
                               flat_time=n_samples * 2e-5, system=system)
        adc = pp.make_adc(num_samples=n_samples, duration=gx.flat_time,
                          delay=gx.rise_time, system=system)
        seq.add_block(rf, gz)
        seq.add_block(gx, adc)
        if shot < n_shots - 1:
            seq.add_block(pp.make_delay(5e-3))   # recovery gap between shots

    seq.set_definition('B0', b0_tesla)
    seq.set_definition('FOV', [fov_m, fov_m, fov_m])
    seq.write(path)
