# Verification record

On 2026-10-31, the standalone package was run against saved predictions from `experiment_native_left_tail_prior_20261030.py`, using the same input grids and software environment. Array comparisons included all pixels, including the NoData code.

| Input | Representation | Accepted sources | Water pixels | Pixels different from saved research mask |
| --- | --- | ---: | ---: | ---: |
| Da Nang | Supplied dB | 106 | 4,495,814 | 0 |
| Assam | Linear power | 108 | 418,109 | 0 |
| Zhengzhou Southwest | Amplitude DN | 382 | 804,562 | 0 |
| Mozambique AOI02 Northeast | Linear power | 239 | 527,092 | 0 |
| Poyang Lake | Linear power | 486 | 3,970,960 | 0 |

The fitted component means, variances and weights, left crossing, profile radius, and local-prior iteration count matched the saved run metadata in all five scenes. In Zhengzhou Southwest no qualifying left crossing was found. The grid transform and CRS matched as well. These checks establish equivalence only for the checked cases and environment, not predictive accuracy, independent validation, or universal cross-platform bitwise equivalence. The images and saved research masks are not included in this repository.

The source contrast gate uses the selected, rebuilt ordered profiles, matching the current 0.15 research implementation. The experiment's prediction did not read truth data. Comparison was to previously saved algorithm output, not to a truth mask.
