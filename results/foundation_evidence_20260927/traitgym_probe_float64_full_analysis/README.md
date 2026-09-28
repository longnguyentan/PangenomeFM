# TraitGym frozen locus-prior results

Exploratory follow-up on previously inspected TraitGym variants, using all five manuscript chromosome folds and all seeds/contexts. Improvement of a downstream classifier does not establish better encoder weights. V is an author-provided NT-2.5B allele score; S remains frozen NT-v2 50M. HGB is a fixed 150-tree-budget estimator without a convergence criterion; only logistic probes can be called converged. No official leaderboard equivalence.

Completed 60 declared runs; 1500 evaluations. All linear probes converged and all fixed-budget fits completed.

## All primary and handcrafted-control AUPRC contrasts

| Dataset | Context | Comparison | ΔAP | 95% CI | Fold p | BH q |
|---|---|---|---:|---|---:|---:|
| complex_traits | 1hop | T_given_CSH_linear | -0.001674 | [-0.006710, +0.002837] | 0.6875 | 0.9625 |
| complex_traits | 1hop | T_given_CSVH_histgb | +0.000705 | [-0.002233, +0.003746] | 0.5000 | 0.8485 |
| complex_traits | 1hop | T_given_CSVH_linear | +0.001072 | [-0.001180, +0.003406] | 0.1875 | 0.8485 |
| complex_traits | 1hop | T_given_CSV_histgb | -0.003519 | [-0.010137, +0.002225] | 0.4375 | 0.8485 |
| complex_traits | 1hop | T_given_CSV_linear | -0.000360 | [-0.003955, +0.003019] | 0.9375 | 1.0000 |
| complex_traits | 1hop | T_given_CS_linear | +0.000753 | [-0.002269, +0.004003] | 0.7500 | 0.9767 |
| complex_traits | 1hop | T_given_V_CSH_fusion | -0.002238 | [-0.006282, +0.001656] | 0.1875 | 0.8485 |
| complex_traits | 1hop | T_given_V_CS_fusion | -0.001132 | [-0.005822, +0.002839] | 0.6250 | 0.9091 |
| complex_traits | 1hop | fusion_vs_V_CS | +0.009795 | [+0.000570, +0.019020] | 0.2500 | 0.8485 |
| complex_traits | 1hop | fusion_vs_V_CSH | +0.007954 | [+0.000418, +0.016547] | 0.1875 | 0.8485 |
| complex_traits | 1hop | fusion_vs_V_CSHT | +0.005716 | [-0.002880, +0.014714] | 0.3750 | 0.8485 |
| complex_traits | 1hop | fusion_vs_V_CST | +0.008663 | [+0.000039, +0.017145] | 0.1875 | 0.8485 |
| complex_traits | 1hop | fusion_vs_concat_CSV | -0.001554 | [-0.007290, +0.002395] | 0.8125 | 0.9891 |
| complex_traits | 1hop | fusion_vs_concat_CSVH | -0.000666 | [-0.002697, +0.000993] | 0.6875 | 0.9625 |
| complex_traits | 1hop | fusion_vs_concat_CSVHT | -0.003976 | [-0.007503, -0.000610] | 0.1250 | 0.8485 |
| complex_traits | 1hop | fusion_vs_concat_CSVT | -0.002327 | [-0.005323, -0.000010] | 0.1250 | 0.8485 |
| complex_traits | 1hop | histgb_vs_fixed_CS | -0.004764 | [-0.012376, +0.001782] | 0.3750 | 0.8485 |
| complex_traits | 1hop | histgb_vs_fixed_CSV | -0.001043 | [-0.011945, +0.006848] | 0.9375 | 1.0000 |
| complex_traits | 1hop | histgb_vs_fixed_CSVH | -0.001722 | [-0.012831, +0.006496] | 0.8750 | 0.9899 |
| complex_traits | 1hop | histgb_vs_fixed_CSVHT | -0.000541 | [-0.010729, +0.008295] | 0.9375 | 1.0000 |
| complex_traits | 1hop | histgb_vs_fixed_CSVT | -0.003508 | [-0.012680, +0.004604] | 0.5625 | 0.8750 |
| complex_traits | 1hop | histgb_vs_fixed_V | -0.001813 | [-0.013549, +0.010270] | 0.8125 | 0.9891 |
| complex_traits | 1hop | linear_vs_fixed_CS | -0.000073 | [-0.005150, +0.004468] | 1.0000 | 1.0000 |
| complex_traits | 1hop | linear_vs_fixed_CSV | +0.002046 | [-0.001440, +0.004968] | 0.3750 | 0.8485 |
| complex_traits | 1hop | linear_vs_fixed_CSVH | +0.001669 | [-0.001710, +0.005048] | 0.4375 | 0.8485 |
| complex_traits | 1hop | linear_vs_fixed_CSVHT | +0.003217 | [+0.000741, +0.005954] | 0.1250 | 0.8485 |
| complex_traits | 1hop | linear_vs_fixed_CSVT | +0.002740 | [+0.000405, +0.005172] | 0.0625 | 0.8485 |
| complex_traits | 1hop | linear_vs_fixed_V | +0.000350 | [+0.000002, +0.000699] | 0.2500 | 0.8485 |
| complex_traits | strict | T_given_CSH_linear | -0.002515 | [-0.005414, -0.000138] | 0.1250 | 0.8485 |
| complex_traits | strict | T_given_CSVH_histgb | +0.000211 | [-0.005515, +0.003981] | 0.9375 | 1.0000 |
| complex_traits | strict | T_given_CSVH_linear | -0.002386 | [-0.005639, +0.000182] | 0.0625 | 0.8485 |
| complex_traits | strict | T_given_CSV_histgb | -0.004974 | [-0.012800, +0.000829] | 0.4375 | 0.8485 |
| complex_traits | strict | T_given_CSV_linear | -0.001796 | [-0.005961, +0.002289] | 0.5000 | 0.8485 |
| complex_traits | strict | T_given_CS_linear | -0.001907 | [-0.004723, +0.001020] | 0.3125 | 0.8485 |
| complex_traits | strict | T_given_V_CSH_fusion | -0.005407 | [-0.009755, -0.001773] | 0.0625 | 0.8485 |
| complex_traits | strict | T_given_V_CS_fusion | -0.005823 | [-0.012923, -0.000058] | 0.2500 | 0.8485 |
| complex_traits | strict | fusion_vs_V_CS | +0.009795 | [+0.000570, +0.019020] | 0.2500 | 0.8485 |
| complex_traits | strict | fusion_vs_V_CSH | +0.007954 | [+0.000418, +0.016547] | 0.1875 | 0.8485 |
| complex_traits | strict | fusion_vs_V_CSHT | +0.002547 | [-0.002898, +0.009513] | 0.7500 | 0.9767 |
| complex_traits | strict | fusion_vs_V_CST | +0.003971 | [-0.001904, +0.011735] | 0.5000 | 0.8485 |
| complex_traits | strict | fusion_vs_concat_CSV | -0.001554 | [-0.007290, +0.002395] | 0.8125 | 0.9891 |
| complex_traits | strict | fusion_vs_concat_CSVH | -0.000666 | [-0.002697, +0.000993] | 0.6875 | 0.9625 |
| complex_traits | strict | fusion_vs_concat_CSVHT | -0.003687 | [-0.010561, +0.000503] | 0.5000 | 0.8485 |
| complex_traits | strict | fusion_vs_concat_CSVT | -0.005582 | [-0.011721, -0.001104] | 0.0625 | 0.8485 |
| complex_traits | strict | histgb_vs_fixed_CS | -0.004764 | [-0.012376, +0.001782] | 0.3750 | 0.8485 |
| complex_traits | strict | histgb_vs_fixed_CSV | -0.001043 | [-0.011945, +0.006848] | 0.9375 | 1.0000 |
| complex_traits | strict | histgb_vs_fixed_CSVH | -0.001722 | [-0.012831, +0.006496] | 0.8750 | 0.9899 |
| complex_traits | strict | histgb_vs_fixed_CSVHT | +0.000264 | [-0.007867, +0.007624] | 0.8125 | 0.9891 |
| complex_traits | strict | histgb_vs_fixed_CSVT | -0.003844 | [-0.010681, +0.003133] | 0.4375 | 0.8485 |
| complex_traits | strict | histgb_vs_fixed_V | -0.001813 | [-0.013549, +0.010270] | 0.8125 | 0.9891 |
| complex_traits | strict | linear_vs_fixed_CS | -0.000073 | [-0.005150, +0.004468] | 1.0000 | 1.0000 |
| complex_traits | strict | linear_vs_fixed_CSV | +0.002046 | [-0.001440, +0.004968] | 0.3750 | 0.8485 |
| complex_traits | strict | linear_vs_fixed_CSVH | +0.001669 | [-0.001710, +0.005048] | 0.4375 | 0.8485 |
| complex_traits | strict | linear_vs_fixed_CSVHT | +0.001058 | [-0.001796, +0.004606] | 1.0000 | 1.0000 |
| complex_traits | strict | linear_vs_fixed_CSVT | +0.002423 | [-0.000569, +0.005596] | 0.2500 | 0.8485 |
| complex_traits | strict | linear_vs_fixed_V | +0.000350 | [+0.000002, +0.000699] | 0.2500 | 0.8485 |
| mendelian_traits | 1hop | T_given_CSH_linear | -0.006379 | [-0.019292, +0.002269] | 0.3750 | 0.8485 |
| mendelian_traits | 1hop | T_given_CSVH_histgb | -0.004509 | [-0.026991, +0.025106] | 0.7500 | 0.9767 |
| mendelian_traits | 1hop | T_given_CSVH_linear | -0.007674 | [-0.027955, +0.004630] | 0.6250 | 0.9091 |
| mendelian_traits | 1hop | T_given_CSV_histgb | -0.000218 | [-0.039130, +0.053114] | 1.0000 | 1.0000 |
| mendelian_traits | 1hop | T_given_CSV_linear | -0.003608 | [-0.015347, +0.004527] | 0.7500 | 0.9767 |
| mendelian_traits | 1hop | T_given_CS_linear | -0.004642 | [-0.019640, +0.006289] | 0.7500 | 0.9767 |
| mendelian_traits | 1hop | T_given_V_CSH_fusion | -0.000407 | [-0.004026, +0.002247] | 1.0000 | 1.0000 |
| mendelian_traits | 1hop | T_given_V_CS_fusion | +0.000590 | [-0.000861, +0.002562] | 0.5000 | 0.8485 |
| mendelian_traits | 1hop | fusion_vs_V_CS | -0.027595 | [-0.066464, +0.000000] | 0.5000 | 0.8485 |
| mendelian_traits | 1hop | fusion_vs_V_CSH | -0.023273 | [-0.051275, +0.000000] | 0.5000 | 0.8485 |
| mendelian_traits | 1hop | fusion_vs_V_CSHT | -0.023679 | [-0.051579, +0.000000] | 0.5000 | 0.8485 |
| mendelian_traits | 1hop | fusion_vs_V_CST | -0.027004 | [-0.064548, +0.000000] | 0.5000 | 0.8485 |
| mendelian_traits | 1hop | fusion_vs_concat_CSV | +0.019170 | [-0.029949, +0.080126] | 0.6250 | 0.9091 |
| mendelian_traits | 1hop | fusion_vs_concat_CSVH | +0.026831 | [-0.023286, +0.082266] | 0.5625 | 0.8750 |
| mendelian_traits | 1hop | fusion_vs_concat_CSVHT | +0.034099 | [-0.003160, +0.081663] | 0.2500 | 0.8485 |
| mendelian_traits | 1hop | fusion_vs_concat_CSVT | +0.023369 | [-0.016450, +0.076863] | 0.6250 | 0.9091 |
| mendelian_traits | 1hop | histgb_vs_fixed_CS | +0.071120 | [-0.000316, +0.150165] | 0.1250 | 0.8485 |
| mendelian_traits | 1hop | histgb_vs_fixed_CSV | +0.068946 | [+0.017423, +0.145067] | 0.1250 | 0.8485 |
| mendelian_traits | 1hop | histgb_vs_fixed_CSVH | +0.060369 | [+0.001586, +0.145565] | 0.1250 | 0.8485 |
| mendelian_traits | 1hop | histgb_vs_fixed_CSVHT | +0.049944 | [-0.028330, +0.143284] | 0.3125 | 0.8485 |
| mendelian_traits | 1hop | histgb_vs_fixed_CSVT | +0.063738 | [-0.009702, +0.148206] | 0.3125 | 0.8485 |
| mendelian_traits | 1hop | histgb_vs_fixed_V | -0.037153 | [-0.068393, -0.013941] | 0.0625 | 0.8485 |
| mendelian_traits | 1hop | linear_vs_fixed_CS | +0.010562 | [-0.028454, +0.055952] | 0.8750 | 0.9899 |
| mendelian_traits | 1hop | linear_vs_fixed_CSV | +0.010811 | [-0.014636, +0.039812] | 0.5625 | 0.8750 |
| mendelian_traits | 1hop | linear_vs_fixed_CSVH | +0.003288 | [-0.024064, +0.031127] | 0.8750 | 0.9899 |
| mendelian_traits | 1hop | linear_vs_fixed_CSVHT | -0.010303 | [-0.049642, +0.015228] | 0.9375 | 1.0000 |
| mendelian_traits | 1hop | linear_vs_fixed_CSVT | +0.002213 | [-0.027820, +0.024411] | 0.8125 | 0.9891 |
| mendelian_traits | 1hop | linear_vs_fixed_V | -0.000684 | [-0.001790, +0.000067] | 0.5000 | 0.8485 |
| mendelian_traits | strict | T_given_CSH_linear | -0.008391 | [-0.020174, +0.001208] | 0.3750 | 0.8485 |
| mendelian_traits | strict | T_given_CSVH_histgb | -0.014110 | [-0.058795, +0.019437] | 0.5625 | 0.8750 |
| mendelian_traits | strict | T_given_CSVH_linear | -0.005132 | [-0.016026, +0.003970] | 0.5000 | 0.8485 |
| mendelian_traits | strict | T_given_CSV_histgb | -0.017290 | [-0.050363, +0.007787] | 0.3125 | 0.8485 |
| mendelian_traits | strict | T_given_CSV_linear | -0.006051 | [-0.015594, +0.002327] | 0.5000 | 0.8485 |
| mendelian_traits | strict | T_given_CS_linear | -0.007651 | [-0.020335, +0.003413] | 0.4375 | 0.8485 |
| mendelian_traits | strict | T_given_V_CSH_fusion | +0.002704 | [-0.008296, +0.018984] | 1.0000 | 1.0000 |
| mendelian_traits | strict | T_given_V_CS_fusion | +0.001940 | [-0.009611, +0.015979] | 1.0000 | 1.0000 |
| mendelian_traits | strict | fusion_vs_V_CS | -0.027595 | [-0.066464, +0.000000] | 0.5000 | 0.8485 |
| mendelian_traits | strict | fusion_vs_V_CSH | -0.023273 | [-0.051275, +0.000000] | 0.5000 | 0.8485 |
| mendelian_traits | strict | fusion_vs_V_CSHT | -0.020569 | [-0.046321, +0.000000] | 0.5000 | 0.8485 |
| mendelian_traits | strict | fusion_vs_V_CST | -0.025654 | [-0.055045, +0.000000] | 0.5000 | 0.8485 |
| mendelian_traits | strict | fusion_vs_concat_CSV | +0.019170 | [-0.029949, +0.080126] | 0.6250 | 0.9091 |
| mendelian_traits | strict | fusion_vs_concat_CSVH | +0.026831 | [-0.023286, +0.082266] | 0.5625 | 0.8750 |
| mendelian_traits | strict | fusion_vs_concat_CSVHT | +0.034667 | [-0.011836, +0.088172] | 0.3750 | 0.8485 |
| mendelian_traits | strict | fusion_vs_concat_CSVT | +0.027162 | [-0.017703, +0.082147] | 0.5000 | 0.8485 |
| mendelian_traits | strict | histgb_vs_fixed_CS | +0.071120 | [-0.000316, +0.150165] | 0.1250 | 0.8485 |
| mendelian_traits | strict | histgb_vs_fixed_CSV | +0.068946 | [+0.017423, +0.145067] | 0.1250 | 0.8485 |
| mendelian_traits | strict | histgb_vs_fixed_CSVH | +0.060369 | [+0.001586, +0.145565] | 0.1250 | 0.8485 |
| mendelian_traits | strict | histgb_vs_fixed_CSVHT | +0.041947 | [-0.018752, +0.108361] | 0.3125 | 0.8485 |
| mendelian_traits | strict | histgb_vs_fixed_CSVT | +0.052758 | [+0.001271, +0.118362] | 0.1875 | 0.8485 |
| mendelian_traits | strict | histgb_vs_fixed_V | -0.037153 | [-0.068393, -0.013941] | 0.0625 | 0.8485 |
| mendelian_traits | strict | linear_vs_fixed_CS | +0.010562 | [-0.028454, +0.055952] | 0.8750 | 0.9899 |
| mendelian_traits | strict | linear_vs_fixed_CSV | +0.010811 | [-0.014636, +0.039812] | 0.5625 | 0.8750 |
| mendelian_traits | strict | linear_vs_fixed_CSVH | +0.003288 | [-0.024064, +0.031127] | 0.8750 | 0.9899 |
| mendelian_traits | strict | linear_vs_fixed_CSVHT | -0.006156 | [-0.048801, +0.028193] | 0.8750 | 0.9899 |
| mendelian_traits | strict | linear_vs_fixed_CSVT | +0.005862 | [-0.024036, +0.036826] | 0.7500 | 0.9767 |
| mendelian_traits | strict | linear_vs_fixed_V | -0.000684 | [-0.001790, +0.000067] | 0.5000 | 0.8485 |

Intervals are pointwise hierarchical fold/seed bootstrap. Exact fold sign-flip p-values and BH-adjusted q-values are also supplied; five folds limit inferential resolution.
