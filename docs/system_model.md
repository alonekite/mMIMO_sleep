# System Model

This document describes the physical model behind the RI=1 massive MIMO antenna
panel muting validation in this repository.

## 1. Antenna array

The base-station panel is configured by a single `ArrayConfig`:

```python
ArrayConfig(
    num_subarray_rows=4,       # logical subarray rows
    num_horizontal=8,          # physical columns (also logical columns)
    elements_per_subarray=2,   # vertical physical elements per logical subarray
    num_polarizations=2,
)
```

This yields:

```text
physical rows    = 4 × 2 = 8
physical columns = 8
physical elements = 8 × 8 = 64
total TX ports   = 64 × 2 = 128
```

The port ordering is **polarization-major**: all ports for pol-0 come first,
then all ports for pol-1. Within each polarization, elements are stored in
row-major order with the horizontal index varying fastest:

```text
port = pol * 64 + row * 8 + col
```

Element pattern and polarization follow the Sionna RT convention:

- TX element pattern: TR 38.901
- TX polarization: `cross` (-45° / +45°)
- RX: single dual-polarized isotropic element (`iso` / `VH`), 2 ports

## 2. DFT codebook

The rank-one precoding codebook is an oversampled DFT grid:

- vertical beams `Nv = 8`
- horizontal beams `Nh = 32`
- co-phasing choices `Ni2 = 4`
- total beams `8 × 32 × 4 = 1024`

The Sionna RT convention is used for the beam weights: each beam applies a
single DFT phase taper across the physical ports. The flat beam index orders
indices as `(i12, i11, i2)` with `i2` changing fastest:

```text
beam_index = (i12 * Nh + i11) * Ni2 + i2
```

The reverse mapping is implemented in `mMIMO_sleep.codebook.pmi`.

## 3. PMI indexing

A rank-one PMI is a triple:

- `i11`: horizontal DFT index, `0 ≤ i11 < Nh`
- `i12`: vertical DFT index, `0 ≤ i12 < Nv`
- `i2`: cross-polarization co-phasing index, `0 ≤ i2 < Ni2`

`beam_index` is the flat row of the codebook matrix returned by
`generate_dft_codebook(...)`; it is also the row used in the wideband beam-sweep
power matrix.

`valid_pmi_mask` is computed from the array radiation pattern at a regular
azimuth/elevation grid and used **only as metadata**. The normal-mode beam sweep
always runs over all 1,024 beams, including those flagged as geometrically
invalid for the panel orientation.

## 4. Muting mask

The sleep mode applies a fixed **right-half physical-port muting mask**:

- active columns: `0, 1, ..., num_horizontal/2 - 1`
- muted columns: `num_horizontal/2, ..., num_horizontal - 1`

For `num_horizontal = 8`, columns 0–3 are active and columns 4–7 are muted.
Because each column exists for every row and both polarizations, exactly half of
the 128 TX ports are muted.

The mask is applied elementwise to the selected beam weights:

```text
w_sleep = w_normal * mask
```

**No power re-normalization is applied.** Therefore:

```text
||w_normal||^2 = 1
||w_sleep||^2  = 0.5
```

The 3 dB linear-power reduction (6 dB in voltage-weighted gain) is the
theoretical baseline, but the actual wideband SNR loss also depends on the
directional power distribution of the selected beam and the multipath channel.

## 5. Wideband CFR

For each UE position the Sionna RT `PathSolver` is invoked once with:

- `max_depth = 15`
- `los = True`
- `specular_reflection = True`
- `diffuse_reflection = True`
- `refraction = False`
- `diffraction = False`
- `seed = 36`

The returned paths are converted into a wideband channel frequency response
(CFR) at the active OFDM subcarriers:

```text
FFT size          = 4096
CP length         = 288
subcarrier spacing = 30 kHz
num RB            = 273
subcarriers/RB    = 12
total active      = 3276
left guard        = 410
right guard       = 410
```

The CFR is normalized only by the solver; no additional normalization is applied
to the weights or the channel. Center-frequency and wideband CFRs are produced
from the same paths object. A consistency assertion verifies that the center
subcarrier of the wideband CFR equals the dedicated center-frequency CFR up to
`rtol=1e-4, atol=1e-12`.

## 6. Normal / sleep SNR

The selected PMI is the beam that maximizes the wideband average effective
channel power:

```text
beam* = argmax_b  mean_k sum_rx |H[k] @ w_b|^2
```

The same PMI is used for both normal and sleep. Received power and SNR are
computed in the linear domain and then converted to dB:

```text
P_rx_dbm  = 30 + 10 log10(mean_k P_rx[k])
SNR_db    = 10 log10(mean_k P_rx[k] / P_noise)
loss_db   = normal_snr_db - sleep_snr_db
```

Because the noise floor is identical and only the beamforming weights change,
the power loss and SNR loss are equal. There is no MCS, BLER, ISI, ICI, or
multi-user interference in this model.

## 7. Complex PDP and delay features

Sionna RT 1.2.2 exposes path coefficients as a tuple `(real, imag)`. The full
complex path coefficient is reconstructed as:

```python
a_complex = np.asarray(paths.a[0]) + 1j * np.asarray(paths.a[1])
```

For the selected normal-mode beam `w_normal`, the beamformed PDP is:

```text
g_p = A_p @ w_normal
P_p = sum_rx |g_p|^2
weighted_delay = sum_p P_p tau_p / sum_p P_p
rms_delay_spread = sqrt( sum_p P_p (tau_p - weighted_delay)^2 / sum_p P_p )
```

where `A_p` is the per-path coefficient matrix with shape `[RX ports, TX ports]`.

`power_weighted_delay_s` is a TA-related proxy, not a quantized 3GPP Timing
Advance command. `rms_delay_spread_s` reports the second central moment of the
beamformed delay profile. Both are computed from the normal-mode PDP; sleep-mode
weights are not used for delay statistics.

## 8. Reference frame and scene

- Scene: Ginza (`ginza_1.xml`) at 3.5 GHz
- TX position: `(-122.0, -108.5, 41.0)` m
- TX orientation: `(0, 15/360 * pi, 0)` rad, i.e. **7.5°** around the y-axis
- TX total power: 51.13 dBm
- Noise figure: 5 dB
- Temperature: 290 K
- Material override: ITU concrete, thickness 0.5 m, scattering 0.3, XPD 0.3

The RX is a single dual-polarized isotropic element placed at each UE location
from the provided `rx_1000.pkl` file.
