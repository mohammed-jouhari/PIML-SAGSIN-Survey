# Supplementary Material

**Physics-Informed Machine Learning for Space-Air-Ground-Sea Integrated Networks: A Tutorial and Survey**
M. Jouhari, H. Benaddi, and M. Guizani

This document collects the material that the article refers to as supplementary material: the details of the literature selection, the full settings of the two worked examples, a note on digital twins over intermittent links, and the list of abbreviations. Section, figure, table, and equation numbers refer to the article.

## Contents

- [S1. Literature selection](#s1-literature-selection)
- [S2. Worked example 1: underwater acoustic link](#s2-worked-example-1-underwater-acoustic-link)
- [S3. Worked example 2: seabed-to-satellite relay through a buoy](#s3-worked-example-2-seabed-to-satellite-relay-through-a-buoy)
- [S4. Digital twins over intermittent links](#s4-digital-twins-over-intermittent-links)
- [S5. Additional remarks](#s5-additional-remarks)
- [S6. Abbreviations](#s6-abbreviations)

## S1. Literature selection

### S1.1 From the screening table to the corpus

The screening table (`data/screening_table.csv`) lists the 246 papers and resources that were coded. Of these, 107 were coded as embedding a physical law. The corpus of 102 papers follows from three operations.

| Operation | Papers | Effect |
|---|---|---|
| Coded as embedding a physical law | 107 | |
| Removed by the venue rule (journal in the third or fourth quartile of the SCImago Journal Rank of 2025) | 3 | 104 |
| Recoded as related work | 1 | 103 |
| Conference paper merged with its journal version | 1 | 102 |

The venue rule removed three papers, two on underwater acoustic and optical channels and one hybrid model of rain attenuation on Earth-space links. All three address the physical layer, so the rule empties or fills no cell of the taxonomy (Fig. 2 of the article).

The four preprints of the corpus are Dabiri et al. (2025), Huang et al. (2026, arXiv:2605.23155), Sarhan et al. (2026), and Bian et al. (2025). They were kept because no peer-reviewed version existed at the time of writing.

### S1.2 Measured data

A paper counts as using measured data when part of its training or test data was measured, including measured inputs such as two-line element (TLE) records or Automatic Identification System (AIS) vessel tracks. A paper does not count as using measured data when a simulation only takes real terrain, weather, or ephemerides as input.

### S1.3 Border cases of criterion (iii)

Criterion (iii) keeps a paper when it embeds an identifiable physical law through one of the eight entry points of Section II-B. Papers whose only physics is training data produced by a simulator were judged one by one, and these border cases are listed with their codes in the screening table.

Examples of the border:

- The physics-aware features of Ma et al. (2026) and the measured channel in the forward pass of Xiao et al. (2026) do not embed a law, so both are related work.
- The Thorp-based world model of Mahgoli et al. (2026) embeds a law, so it is counted.
- A conference paper and its journal extension are counted once, as for the Helmholtz-informed radio-map model of Wang et al. (2025, 2026).

### S1.4 Sensitivity of the counts

The counts depend on criterion (iii) at its border.

| Reading of criterion (iii) | Change in the corpus | Change in the taxonomy |
|---|---|---|
| Stricter: pretraining on simulated data alone is not enough | The eight papers whose only code is S leave the corpus | Physical layer of the air segment from 5 to 3 papers, ground twins from 4 to 3; no cell becomes empty |
| Looser: a physics-based system model is enough | The maritime deep reinforcement learning scheme of Prabhashana et al. (2026) enters the corpus | One paper in the cross-segment row |

## S2. Worked example 1: underwater acoustic link

The article gives the channel model (Eq. (5)), the two oceans, the residuals of the physics loss (Eq. (6)), and the results (Fig. 4). The settings below complete Section II-F.

### S2.1 Channel and data

| Item | Value |
|---|---|
| True parameters of ocean A | spreading factor 1.7, absorption scale 1.25, offset 3 dB |
| Ripple of ocean A | 2 sin(2π l / 1.5) exp(−l / 4) dB, with l in km; contained in no model |
| Nominal (textbook) parameters | spreading factor 1.5, absorption scale 1, offset 0 dB, no ripple |
| Ocean B | nodes 10 m below a sea surface with an rms wave height of 0.5 m; the ripple is replaced by the interference of the direct path with the path reflected by the surface (Lloyd's mirror), whose coherent part decreases with the roughness (Jensen et al., 2011); this term lies between −6 and +16 dB |
| Training region | range 0.2 to 5 km, frequency 2 to 25 kHz, uniform sampling |
| Measurement noise | additive Gaussian, standard deviation 1.5 dB |
| Test region | range 0.2 to 10 km, frequency 2 to 40 kHz, 4000 points drawn uniformly, channel without noise |
| Training set sizes | 10, 30, 100, 300 samples |
| Seeds | 5 per configuration |

### S2.2 Models and training

| Model | Settings |
|---|---|
| Network (all learned models) | two hidden layers of 64 units, tanh activations, inputs ln l and f, both scaled to [0, 1] |
| MLP | data loss only; Adam, 3000 steps, learning rate 2e-3 with cosine decay |
| PINN | data loss plus physics loss at 1000 collocation points drawn uniformly over the full domain; the spreading factor and the absorption scale are trained with the network from their nominal values; 2000 steps of Adam followed by up to 400 iterations of L-BFGS |
| Hybrid (nominal) | nominal law plus network correction; Adam, 3000 steps |
| Hybrid (calibrated) | law calibrated either jointly with the network or first by least squares, after which the network learns the remaining residual |
| Calibrated law | linear least-squares fit of the spreading factor, the absorption scale, and the offset, without ripple |
| GP with the law as mean | calibrated law as mean, estimated jointly with the GP by universal kriging; range length scale bounded by 2.5 km |
| GP with a constant mean | data-only counterpart of the previous model |

The loss of the PINN is the mean of ((TL̂ − y)/10)² over the samples plus the mean of (r_l/10)² + (r_f/10)² over the collocation points, with equal weights and both terms in dB scaled by 10. L-BFGS was not used for the MLP and the hybrid model, because it made them unstable with sparse data.

The full experiment takes about 40 min on a free Colab central processing unit.

### S2.3 Check of the pipeline

The PINN was first trained on 300 samples without noise and without the ripple over the full domain. It found a spreading factor of 1.71 and an absorption scale of 1.25, against the true values of 1.7 and 1.25.

### S2.4 Role of the bound on the length scale of the GP

With 30 samples or fewer, the GP with the law as its mean brings no gain over the calibrated law. Without the bound on its range length scale, its error outside the training region rises from 1.8 to 3.6 dB at 30 samples. The bound keeps the GP on local departures from the law, so that it returns to the law outside the data.

## S3. Worked example 2: seabed-to-satellite relay through a buoy

The article gives the scenario (Fig. 5), the model of the practical gateway (Eq. (16)), and the results (Fig. 6). The parameters below complete Section V-D.

### S3.1 Acoustic hop

| Item | Value |
|---|---|
| Depth of the sensor | 100 m |
| Carrier frequency | 12 kHz |
| Transmission loss | Eq. (5) on the slant path, with the true values and the ripple of ocean A |
| Source level | 165 dB re 1 µPa at 1 m |
| Noise bandwidth | 4 kHz |
| Ambient noise | sum of the usual components; the surface term grows with the wind speed w (Stojanovic, 2007) |
| SNR gap of the modem | 9 dB |

### S3.2 Satellite hop

The SNR of the satellite hop, in dB, is

```
SNR_s = P_t + G_g(ε) + G/T + 228.6 − 10 log10(B_s) − FSPL(d(ε)) − q1 / sin(ε) − q2 − L_t
```

with the slant range d(ε) of Eq. (7) and the free-space path loss of Eq. (8).

| Item | Value |
|---|---|
| Satellite altitude, carrier | 600 km, 2 GHz |
| Transmit power P_t | −13 dBW |
| Satellite G/T | −15 dB/K |
| Bandwidth B_s | 10 kHz |
| Excess loss | q1 = 1 dB, q2 = 1.5 dB |
| Buoy G1 (small, tilts with the waves) | G_g = 3 + 10 log10(sin ε) dBi; tilt loss L_t = 10 log10(1 + k_w w² cos² ε), with k_w = 0.0092 s²/m² |
| Buoy G2 (stabilized antenna) | G_g = 8 dBi; L_t = 0 |
| SNR gap of the modem | 3 dB |

The rate of each hop is B log2(1 + SNR/Γ), where Γ is the SNR gap.

### S3.3 Gateways

| Gateway | End-to-end rate |
|---|---|
| Full duplex | R = min(R_a, R_s) |
| Ideal half duplex | R = R_a R_s / (R_a + R_s) |
| Practical half duplex (main case) | R = L_p / (L_p/R_a + L_p/R_s + T_sw), with packets of L_p = 2.048 kbit |

The switching delay is T_sw = T_0 + τ_g (w/w_0)³, with T_0 = 0.1 s, w_0 = 10 m/s, and τ_g equal to 0.3 s for G1 and 0.05 s for G2. This cubic law is contained in no model.

### S3.4 Data and splits

| Item | Value |
|---|---|
| Measured conditions | buoys within 3 km, elevations above 40°, winds of 2 to 8 m/s |
| Snapshot | SNR of each hop, with errors of 1.5 dB (acoustic) and 1 dB (satellite); throughput, with a log-normal error of 5% |
| Training set sizes | 20, 50, 150, 500 snapshots |
| Seeds | 5 per configuration |
| Test: new geometry | buoys up to 6 km, elevations down to 10° |
| Test: storm | winds of 8 to 14 m/s |
| Test: both shifts | new geometry and storm at once |

### S3.5 Models

| Model | Description |
|---|---|
| MLP (data only) | maps the gateway, the distance, the elevation, and the wind to the end-to-end rate |
| Hop MLPs | learn the two SNRs and the delay from the same measurements as the chain |
| Nominal chain | textbook values, no excess or tilt loss, minimum junction |
| Coupled chain | fits the spreading factor, the absorption scale, and the offset of Eq. (5), q1 and q2, and a small-tilt loss c_t w² cos² ε for G1 to the SNRs of each hop by least squares within physical bounds (spreading factor between 1 and 2, other parameters nonnegative) |
| Chain + time sharing without delay | coupled chain with the ideal half-duplex junction |
| Chain + quadratic delay | delay t_0 + t_2 w² per gateway, fitted by bounded least squares |
| Chain + learned delay | delay learned by a small network (hybrid residual at the interface) |

The first-order tilt law of the coupled chain overestimates the loss in a storm: 4.6 dB instead of 3.1 dB at an elevation of 40° and a wind of 14 m/s. The networks have two hidden layers with tanh activations and are trained with Adam. The code and the results of all 13 models and of the three gateways are in `code/`.

### S3.6 Statistical note

A result obtained in 5 of 5 seeds has a two-sided sign-test p-value of 0.0625. Such a result is consistent over the seeds but not statistically significant at the 5% level.

## S4. Digital twins over intermittent links

A twin is useful only as long as it matches the network. In space and under water, the measurements reach the twin rarely and with delay, because the contacts are short and the acoustic links are slow. A measurement over 2 km of acoustic path describes a state that changed 1.3 s ago.

Between two updates, the twin can propagate its state with a physical model, such as a TLE set with SGP4 (Vallado et al., 2006), and recalibrate it when a measurement arrives. The error of a propagator grows with the time since the last update, and a neural ODE was reported to have a lower error than SGP4 over 7 days (Subramanian et al., 2023).

Recalibration with few samples is also uncertain, as the spread of the learned parameters over the seeds in the first worked example shows: with 30 samples, the learned spreading factor and absorption scale of the PINN have a Spearman rank correlation of −0.90 over the seeds. The bandwidth saved by sending only a few parameters of the sound-speed profile must therefore be weighed against the identifiability and the computation that the receiver requires.

## S5. Additional remarks

- **Delay-aware medium access under water.** In the multi-agent protocol of Huang et al. (2025), the throughput can exceed that of a system without propagation delay. Our interpretation of this gain is that packets sent at the same time by nodes at different distances reach the receiver at different times, so the delay creates idle slots.
- **Routing in a maritime network.** The quantum-enhanced deep reinforcement learning scheme of Prabhashana et al. (2026) reached a lower system cost than conventional deep reinforcement learning, with the mobility as part of the system model. It embeds no physical law and is therefore not counted in the corpus.
- **Goal-oriented semantic twins.** A case study of collaborative tracking in satellite and UAV networks showed a better timeliness of the perceptual data than a conventional twin (Qiu et al., 2026).
- **Differentiable ray tracing.** Differentiable ray tracers, such as Sionna RT, compute the gradients of the channel with respect to material properties and positions (Hoydis et al., 2023), but they need an accurate geometry and accurate material properties.
- **Simulated optical channels.** A network trained on a wave-optics simulation driven by Argo float data predicted the received optical intensity with an accuracy above 98% and a runtime reduction of more than 90% (Xu et al., 2026). It is related work, since the physics enters only through the training data.
- **Attacks on physics-informed models.** A PINN-based state estimator for power systems was up to about 93% more accurate than an equivalent neural network under data manipulation (Falas et al., 2025). The transfer of this result to SAGSIN has not been tested.
- **General PDE benchmarks.** PDEBench provides PDE datasets and baselines such as FNO, U-Net, and PINN, but its tasks contain no SAGSIN medium (Takamoto et al., 2022).
- **Audit against standards.** A model that corrects a standard model, such as the 3GPP NTN channel models (TR 38.811, TR 38.821) or the ITU-R recommendations (P.618, P.676, P.840), can be audited against the standard that operators already trust.
- **Earlier air-to-ground campaigns.** The measurement campaigns before 2019 are surveyed by Khawaja et al. (2019).

## S6. Abbreviations

| Abbreviation | Meaning |
|---|---|
| 3D | Three-dimensional |
| 3GPP | 3rd Generation Partnership Project |
| 6G | Sixth generation |
| A2G | Air-to-ground |
| AI | Artificial intelligence |
| AIS | Automatic Identification System |
| AUC | Area under the ROC curve |
| AUV | Autonomous underwater vehicle |
| CFD | Computational fluid dynamics |
| CMDP | Constrained Markov decision process |
| CNN | Convolutional neural network |
| CSI | Channel state information |
| CUSUM | Cumulative sum |
| DeepONet | Deep operator network |
| DRL | Deep reinforcement learning |
| DVL | Doppler velocity log |
| EDH | Evaporation duct height |
| ERA5 | Fifth-generation reanalysis of the European Centre for Medium-Range Weather Forecasts |
| FNO | Fourier neural operator |
| FSPL | Free-space path loss |
| GEO | Geostationary Earth orbit |
| GNSS | Global navigation satellite system |
| GP | Gaussian process |
| GRU | Gated recurrent unit |
| HAPS | High-altitude platform station |
| ITU-R | International Telecommunication Union Radiocommunication Sector |
| L-BFGS | Limited-memory Broyden--Fletcher--Goldfarb--Shanno |
| LEO | Low Earth orbit |
| LoS | Line of sight |
| LS | Least squares |
| LSTM | Long short-term memory |
| MAE | Mean absolute error |
| MEO | Medium Earth orbit |
| MIMO | Multiple-input multiple-output |
| MLP | Multilayer perceptron |
| NLoS | Non-line of sight |
| NMSE | Normalized mean square error |
| NTN | Non-terrestrial network |
| ODE | Ordinary differential equation |
| PDE | Partial differential equation |
| PE | Parabolic equation |
| PIDRTN | Physics-informed deep ray tracing network |
| PIML | Physics-informed machine learning |
| PINN | Physics-informed neural network |
| PL | Path loss |
| RIS | Reconfigurable intelligent surface |
| RL | Reinforcement learning |
| RMSE | Root mean square error |
| ROC | Receiver operating characteristic |
| SAGIN | Space-air-ground integrated network |
| SAGSIN | Space-air-ground-sea integrated network |
| SGP4 | Simplified General Perturbations 4 |
| SNR | Signal-to-noise ratio |
| SSP | Sound-speed profile |
| TARF | Translational acoustic-radio frequency |
| TL | Transmission loss |
| TLE | Two-line element |
| UAV | Unmanned aerial vehicle |
| USV | Unmanned surface vehicle |
| VAE | Variational autoencoder |
