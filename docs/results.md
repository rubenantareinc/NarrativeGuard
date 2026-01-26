# Results & Analysis

## Ablation summary
| Feature set | MAE | RMSE | R² |
| --- | --- | --- | --- |
| structured_only | 4.950899447817316 | 5.911748150164717 | 0.2440319764114368 |
| text_only | 5.266851388233674 | 6.618965001071727 | 0.05234175191486434 |
| combined | 4.811011541920891 | 5.828704100851181 | 0.2651214102909344 |

## Interpretation
- Structured indicators explain a meaningful share of variance, but the overall fit remains modest.
- Text-only performance is weaker, suggesting sparse or noisy coverage for some city-months.
- Combining text with structured features yields a small but consistent lift in MAE and RMSE.
- The R² gains are incremental, reinforcing that text signals supplement rather than replace structured indicators.

## Top features (combined model)
Top features by absolute coefficient will be added after the next run.
