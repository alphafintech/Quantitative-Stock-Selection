# Quantitative-Stock-Selection
Quantitative Stock Selection

The S&P 500 symbol list is no longer downloaded when the finance module is
imported.  Instead call ``load_sp500_meta()`` from
``GPT.compute_high_growth_score_SP500_GPT`` to fetch it lazily when first
needed.
