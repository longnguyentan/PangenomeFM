# Validation-only genotyping-quality fallback

Exploratory follow-up; no encoders or probes retrained. Positive gains mean lower error.

| Context | Contrast | MAE reduction | 95% CI |
|---|---|---:|---|
| 1hop | fallback_T_given_CS | +0.000969 | [-0.001362, +0.004661] |
| 1hop | fallback_T_given_CSH | -0.000222 | [-0.004284, +0.004055] |
| 1hop | fallback_T_given_CSL | +0.001817 | [-0.001315, +0.006800] |
| 1hop | fallback_T_given_CSLH | +0.001130 | [-0.000816, +0.004251] |
| strict | fallback_T_given_CS | +0.000250 | [+0.000000, +0.000662] |
| strict | fallback_T_given_CSH | +0.000146 | [-0.000047, +0.000449] |
| strict | fallback_T_given_CSL | +0.000241 | [+0.000000, +0.000634] |
| strict | fallback_T_given_CSLH | +0.001472 | [+0.000000, +0.004151] |
