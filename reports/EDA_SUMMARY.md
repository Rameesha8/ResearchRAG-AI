# EDA Summary

## Current Dataset Snapshot

- Source corpus: arXiv metadata snapshot
- Processed records used for retrieval: 5,000
- Unique primary categories in processed subset: 122
- Average title length: 70.43 characters
- Average abstract length: 802.50 characters

## Top 10 Primary Categories

1. astro-ph: 972
2. hep-ph: 326
3. hep-th: 305
4. quant-ph: 257
5. gr-qc: 166
6. cond-mat.mtrl-sci: 164
7. cond-mat.mes-hall: 151
8. cond-mat.str-el: 150
9. cond-mat.stat-mech: 135
10. math.AG: 114

## Key Insights

- The processed arXiv subset is imbalanced, with `astro-ph` dominating the sample.
- Title and abstract text are sufficiently rich for text classification and retrieval tasks.
- Because of class imbalance, macro-averaged metrics are important for fair model comparison.
- The top-category setup used for classification reduces label sparsity and makes the fellowship model-comparison task practical.
