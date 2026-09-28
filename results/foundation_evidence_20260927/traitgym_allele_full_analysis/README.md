# TraitGym frozen locus-prior results

Five-fold locus-prior adaptation, not the official leave-one-chromosome-out TraitGym leaderboard. Static T and S cannot distinguish alternate alleles at the same segment. Favorable T gains do not establish learned-weights superiority without random-encoder controls. Report all null and negative outcomes. V is an author-provided allele-aware sequence score, not newly trained PangenomeFM or a reproduction of the complete official NT benchmark. Previously observed test results motivated this sensitivity; any gain is exploratory.

Completed 60 declared runs; 480 fits. All converged: True.

## All primary and handcrafted-control AUPRC contrasts

| Dataset | Context | Comparison | ΔAP | 95% CI | Fold p | BH q |
|---|---|---|---:|---|---:|---:|
| complex_traits | 1hop | SV_given_CT | +0.008464 | [-0.000071, +0.017143] | 0.2500 | 0.5556 |
| complex_traits | 1hop | T_given_CSV | -0.001054 | [-0.004027, +0.001481] | 0.3750 | 0.6000 |
| complex_traits | 1hop | T_given_CSVH | -0.000476 | [-0.002651, +0.001401] | 0.6875 | 0.8667 |
| complex_traits | 1hop | V_given_CS | +0.000493 | [-0.000429, +0.001362] | 0.3125 | 0.5556 |
| complex_traits | strict | SV_given_CT | +0.011168 | [+0.004406, +0.018204] | 0.0625 | 0.3333 |
| complex_traits | strict | T_given_CSV | -0.002173 | [-0.004831, +0.000003] | 0.1875 | 0.5000 |
| complex_traits | strict | T_given_CSVH | -0.001775 | [-0.003740, +0.000143] | 0.1250 | 0.4000 |
| complex_traits | strict | V_given_CS | +0.000493 | [-0.000429, +0.001362] | 0.3125 | 0.5556 |
| mendelian_traits | 1hop | SV_given_CT | +0.004106 | [-0.025369, +0.032069] | 1.0000 | 1.0000 |
| mendelian_traits | 1hop | T_given_CSV | +0.004990 | [-0.010788, +0.021374] | 0.6250 | 0.8667 |
| mendelian_traits | 1hop | T_given_CSVH | +0.005916 | [-0.010227, +0.027539] | 0.7500 | 0.8667 |
| mendelian_traits | 1hop | V_given_CS | -0.018730 | [-0.028974, -0.008670] | 0.0625 | 0.3333 |
| mendelian_traits | strict | SV_given_CT | -0.071948 | [-0.144123, -0.012779] | 0.1250 | 0.4000 |
| mendelian_traits | strict | T_given_CSV | -0.001102 | [-0.012369, +0.011257] | 0.8125 | 0.8667 |
| mendelian_traits | strict | T_given_CSVH | +0.004312 | [-0.011835, +0.028005] | 0.8125 | 0.8667 |
| mendelian_traits | strict | V_given_CS | -0.018730 | [-0.028974, -0.008670] | 0.0625 | 0.3333 |

Intervals are pointwise hierarchical fold/seed bootstrap. Exact fold sign-flip p-values and BH-adjusted q-values are also supplied; five folds limit inferential resolution.
