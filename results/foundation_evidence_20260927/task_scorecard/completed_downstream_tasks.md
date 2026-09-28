# Completed downstream task scorecard

AP values and paired 95% intervals come from the indicated saved source tables. Intervals use the source hierarchical fold/seed bootstrap. These are not independent tests of architecture choices made after viewing these results. Positive intervals are not equivalent to multiplicity-adjusted significance.

C = manuscript coordinate/structural features; S = frozen NT; T = frozen topology v1. Rows marked ENCODE subtype/complexity are subsets of the original binary predictions.

| Task | Context | C+S AP | C+S+T AP | ΔT [95% CI] | Runs |
|---|---|---:|---:|---:|---:|
| ENCODE binary cCRE | strict | 0.918458 | 0.922466 | +0.004008 [+0.002930, +0.004804] | 15 |
| ENCODE binary cCRE | 1hop | 0.918458 | 0.921865 | +0.003406 [+0.001653, +0.004797] | 15 |
| HGSVC3 INS versus DEL | strict | 0.872084 | 0.906330 | +0.034246 [+0.032396, +0.036228] | 15 |
| HGSVC3 INS versus DEL | 1hop | 0.872084 | 0.912174 | +0.040090 [+0.036086, +0.043957] | 15 |
| ENCODE CTCF-only | 1hop | 0.399250 | 0.447452 | +0.048203 [+0.034248, +0.061313] | 15 |
| ENCODE CTCF-only | strict | 0.399235 | 0.434133 | +0.034898 [+0.024345, +0.043586] | 15 |
| ENCODE PLS | 1hop | 0.051054 | 0.057070 | +0.006016 [+0.000961, +0.010552] | 15 |
| ENCODE PLS | strict | 0.051051 | 0.056827 | +0.005776 [+0.002405, +0.008517] | 15 |
| ENCODE dELS | 1hop | 0.917038 | 0.921668 | +0.004629 [+0.001995, +0.006712] | 15 |
| ENCODE dELS | strict | 0.917036 | 0.922403 | +0.005367 [+0.003695, +0.006545] | 15 |
| ENCODE pELS | 1hop | 0.607739 | 0.633299 | +0.025560 [+0.013555, +0.035123] | 15 |
| ENCODE pELS | strict | 0.607733 | 0.628663 | +0.020930 [+0.013053, +0.026293] | 15 |
| ENCODE high | 1hop | 0.863928 | 0.869279 | +0.005351 [+0.002255, +0.008173] | 15 |
| ENCODE high | strict | 0.863924 | 0.871500 | +0.007576 [+0.006397, +0.008616] | 15 |
| ENCODE low | 1hop | 0.967820 | 0.969631 | +0.001811 [+0.001104, +0.002702] | 15 |
| ENCODE low | strict | 0.967825 | 0.969575 | +0.001750 [+0.001090, +0.002383] | 15 |
| ENCODE medium | 1hop | 0.957610 | 0.959600 | +0.001990 [+0.000892, +0.003113] | 15 |
| ENCODE medium | strict | 0.957611 | 0.959020 | +0.001409 [+0.000585, +0.002252] | 15 |
| P0 AS-prone cCRE | 1hop | 0.151416 | 0.150762 | -0.000654 [-0.002092, +0.000466] | 15 |
| P0 AS-prone cCRE | strict | 0.151416 | 0.151625 | +0.000208 [-0.000317, +0.000746] | 15 |
| P0 exposure matched | 1hop | 0.533617 | 0.535618 | +0.002002 [+0.000844, +0.003278] | 15 |
| P0 exposure matched | strict | 0.533617 | 0.535099 | +0.001482 [+0.000098, +0.002925] | 15 |
| P0 H3K27ac only | 1hop | 0.062326 | 0.062938 | +0.000612 [-0.000533, +0.001689] | 15 |
| P0 H3K27ac only | strict | 0.062326 | 0.062910 | +0.000584 [+0.000084, +0.001065] | 15 |
| P0 CTCF only | 1hop | 0.078094 | 0.078819 | +0.000725 [+0.000217, +0.001301] | 15 |
| P0 CTCF only | strict | 0.078094 | 0.078354 | +0.000261 [-0.000370, +0.000952] | 15 |
| P1 enhancer tissue macro | 1hop | 0.569931 | 0.570735 | +0.000805 [-0.001531, +0.002467] | 15 |
| P1 enhancer tissue macro | strict | 0.569931 | 0.570820 | +0.000889 [-0.000389, +0.002326] | 15 |
| P2 CTCF SNV | 1hop | 0.061559 | 0.063805 | +0.002245 [+0.000958, +0.003482] | 15 |
| P2 CTCF SNV | strict | 0.061559 | 0.063033 | +0.001473 [+0.000096, +0.003549] | 15 |
| P2 H3K27ac SNV | 1hop | 0.049680 | 0.053018 | +0.003338 [+0.000120, +0.007962] | 15 |
| P2 H3K27ac SNV | strict | 0.049680 | 0.050373 | +0.000693 [-0.000567, +0.001920] | 15 |
| RNA ASE SNV | 1hop | 0.025530 | 0.026996 | +0.001465 [-0.000218, +0.004303] | 15 |
| RNA ASE SNV | strict | 0.025530 | 0.027099 | +0.001568 [-0.000075, +0.004401] | 15 |
| P2 ATAC SNV | 1hop | 0.047280 | 0.047599 | +0.000319 [-0.000672, +0.001780] | 15 |
| P2 ATAC SNV | strict | 0.047280 | 0.046971 | -0.000309 [-0.000714, +0.000020] | 15 |
| P2 H3K4ME3 SNV | 1hop | 0.080250 | 0.089251 | +0.009000 [+0.000618, +0.018619] | 15 |
| P2 H3K4ME3 SNV | strict | 0.080250 | 0.084671 | +0.004420 [+0.000116, +0.010614] | 15 |
| P2 H3K27ME3 SNV | 1hop | 0.091497 | 0.098773 | +0.007276 [-0.001910, +0.021325] | 15 |
| P2 H3K27ME3 SNV | strict | 0.091497 | 0.094883 | +0.003386 [-0.000456, +0.009559] | 15 |
| P1 enhancer Peyers_patch | 1hop | 0.565275 | 0.565183 | -0.000092 [-0.001569, +0.001113] | 15 |
| P1 enhancer Peyers_patch | strict | 0.565275 | 0.564497 | -0.000778 [-0.001683, +0.000126] | 15 |
| P1 enhancer body_of_pancreas | 1hop | 0.556333 | 0.556806 | +0.000474 [-0.000445, +0.001361] | 15 |
| P1 enhancer body_of_pancreas | strict | 0.556333 | 0.557195 | +0.000862 [-0.000206, +0.002003] | 15 |
| P1 enhancer gastroesophageal_sphincter | 1hop | 0.571966 | 0.572568 | +0.000603 [-0.004254, +0.004279] | 15 |
| P1 enhancer gastroesophageal_sphincter | strict | 0.571966 | 0.572458 | +0.000492 [-0.001782, +0.002691] | 15 |
| P1 enhancer thyroid_gland | 1hop | 0.571855 | 0.573284 | +0.001430 [-0.000742, +0.003184] | 15 |
| P1 enhancer thyroid_gland | strict | 0.571855 | 0.572962 | +0.001107 [-0.000376, +0.002996] | 15 |
| P1 enhancer tibial_nerve | 1hop | 0.584226 | 0.585834 | +0.001608 [-0.002043, +0.004163] | 15 |
| P1 enhancer tibial_nerve | strict | 0.584226 | 0.586990 | +0.002764 [+0.000573, +0.005121] | 15 |
| HG008 clonal INS versus DEL | 1hop | 0.426009 | 0.490872 | +0.064863 [-0.066739, +0.212115] | 15 |
| HG008 clonal INS versus DEL | strict | 0.426009 | 0.458534 | +0.032525 [-0.053371, +0.110254] | 15 |

Source paths and analysis scope are retained in completed_downstream_tasks.csv. Development-only v2/NT/composite results and unresolved tasks are documented separately; no incomplete task receives a fabricated score.
