# Measured genotyping quality: held-out regression

Positive gain means lower MAE after adding T.

| Target | Context | Contrast | MAE gain | 95% CI | Fold p |
|---|---|---|---:|---|---:|
| mean_qv_fraction | 1hop | T_given_CS | -0.000473 | [-0.001633, +0.000676] | 0.5000 |
| mean_qv_fraction | 1hop | T_given_CSH | -0.001401 | [-0.004623, +0.000692] | 0.6250 |
| mean_qv_fraction | 1hop | T_given_CSL | -0.000481 | [-0.001593, +0.000627] | 0.4375 |
| mean_qv_fraction | 1hop | T_given_CSLH | -0.000032 | [-0.001249, +0.001134] | 1.0000 |
| mean_qv_fraction | strict | T_given_CS | +0.000651 | [-0.000202, +0.001641] | 0.3125 |
| mean_qv_fraction | strict | T_given_CSH | -0.000227 | [-0.001796, +0.000896] | 0.7500 |
| mean_qv_fraction | strict | T_given_CSL | +0.000613 | [-0.000256, +0.001625] | 0.3125 |
| mean_qv_fraction | strict | T_given_CSLH | +0.001079 | [-0.001508, +0.004154] | 0.6875 |
| mean_qv_pred | 1hop | T_given_CS | -0.043894 | [-0.156905, +0.027397] | 0.5000 |
| mean_qv_pred | 1hop | T_given_CSH | -0.034800 | [-0.175148, +0.041674] | 1.0000 |
| mean_qv_pred | 1hop | T_given_CSL | -0.037380 | [-0.145844, +0.032390] | 0.5625 |
| mean_qv_pred | 1hop | T_given_CSLH | -0.041037 | [-0.161648, +0.027109] | 0.9375 |
| mean_qv_pred | strict | T_given_CS | -0.000210 | [-0.123486, +0.083923] | 1.0000 |
| mean_qv_pred | strict | T_given_CSH | +0.003644 | [-0.114616, +0.073140] | 1.0000 |
| mean_qv_pred | strict | T_given_CSL | -0.026824 | [-0.161515, +0.080901] | 0.8125 |
| mean_qv_pred | strict | T_given_CSLH | -0.004268 | [-0.106921, +0.063484] | 1.0000 |
