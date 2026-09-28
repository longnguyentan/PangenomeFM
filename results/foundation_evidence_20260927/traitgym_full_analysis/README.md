# TraitGym frozen locus-prior results

Five-fold locus-prior adaptation, not the official leave-one-chromosome-out TraitGym leaderboard. Static T and S cannot distinguish alternate alleles at the same segment. Favorable T gains do not establish learned-weights superiority without random-encoder controls. Report all null and negative outcomes.

Completed 60 declared runs; 540 fits. All converged: True.

## All primary and handcrafted-control AUPRC contrasts

| Dataset | Context | Comparison | ΔAP | 95% CI | Fold p | BH q |
|---|---|---|---:|---|---:|---:|
| complex_traits | 1hop | S_given_CT | +0.008037 | [-0.000847, +0.017149] | 0.2500 | 0.6000 |
| complex_traits | 1hop | T_given_CS | -0.000987 | [-0.003610, +0.001404] | 0.3750 | 0.6429 |
| complex_traits | 1hop | T_given_CSH | -0.000608 | [-0.002669, +0.001216] | 0.6250 | 0.7500 |
| complex_traits | strict | S_given_CT | +0.010823 | [+0.003458, +0.018508] | 0.1250 | 0.5000 |
| complex_traits | strict | T_given_CS | -0.002025 | [-0.004411, -0.000102] | 0.1250 | 0.5000 |
| complex_traits | strict | T_given_CSH | -0.001735 | [-0.003590, +0.000052] | 0.1250 | 0.5000 |
| mendelian_traits | 1hop | S_given_CT | +0.021800 | [-0.010977, +0.055716] | 0.3750 | 0.6429 |
| mendelian_traits | 1hop | T_given_CS | +0.003954 | [-0.016815, +0.021650] | 0.7500 | 0.8182 |
| mendelian_traits | 1hop | T_given_CSH | +0.003523 | [-0.015026, +0.022994] | 0.6250 | 0.7500 |
| mendelian_traits | strict | S_given_CT | -0.057184 | [-0.129840, +0.000468] | 0.1875 | 0.5625 |
| mendelian_traits | strict | T_given_CS | -0.005069 | [-0.017554, +0.007397] | 0.4375 | 0.6562 |
| mendelian_traits | strict | T_given_CSH | +0.000458 | [-0.015901, +0.022022] | 0.8750 | 0.8750 |

Intervals are pointwise hierarchical fold/seed bootstrap. Exact fold sign-flip p-values and BH-adjusted q-values are also supplied; five folds limit inferential resolution.
