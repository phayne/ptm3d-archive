# Haworth permanently shadowed floor: history of the comparison

This page records how the Haworth comparison reached its current state. The current numbers
are in [VALIDATION.md](VALIDATION.md) (Figs. 8 and 9).

- **Earlier comparison (2026, before the revision).** At the central permanently shadowed
  floor of Haworth (87.45°S, 2.1°W), the model's summer-maximum temperature was about 45 K
  against about 65 K from Diviner (Williams et al. 2019), a cold bias of about 20 K. An
  H-matrix defect that zeroed floor-to-wall coupling above about 100,000 faces had been found
  and fixed before that comparison.
- **What changed.** The revision rebuilt the view factors with a Monte Carlo estimator on the
  correct surface side and added sub-facet roughness (beaming), which is appropriate on this
  240 m site mesh.
- **Current result.** At the Haworth SE-rim site, over 2,970 cold-trap cells, the median
  summer-maximum bias is −0.80 K (IQR 11.19 K) with beaming and −10.30 K (IQR 10.38 K)
  without it. The view factors and the sub-facet emission model, not a missing rim heat source,
  account for the earlier bias.
