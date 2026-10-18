# Modeling and engineering choices

The original assessment is available in `docs/assignment.pdf`. The implementation
prioritizes auditable time alignment and out-of-sample validation over complex models.

- **Daily sessions:** sufficient for the assignment's unspecified period n; an
  exchange calendar handles holidays, early closes and DST. Six forecasts refer
  to six unknown future closes, with the known reference close reported separately.
- **Fixed sentiment lexicon:** reuses the project's rules and adds explicit finance
  vocabulary. It avoids training a text scorer using future labels. This is a
  baseline with limitations for sarcasm, ambiguity and financial context.
- **Ridge regression on log returns:** a regularized baseline for a relatively small
  180-calendar-day sample. A fold-local StandardScaler prevents future distribution
  information entering training. Hyperparameters are fixed, not optimized on OOS results.
- **Purged walk-forward folds:** every training label, including the sixth horizon,
  is fully known before the test block. Test origins do not repeat across folds.
- **Two baselines:** last-price persistence checks whether complexity helps, while
  price-only regression measures the incremental contribution of X sentiment.
- **Honest execution timing:** decisions after one close cannot trade at that
  already observed close. The illustrative strategy enters next open and exits
  that session's close, with per-side costs and no stacked multi-day bets.
- **SQLite and snapshots:** raw records, aggregated joins and run provenance remain
  inspectable. No external database is needed to reproduce the assessment locally.
- **News as context:** the brief asks for price/sentiment-only predictors and recent
  news as a complement. Current Tavily headlines belong in the outlook; injecting
  them into historical training would introduce look-ahead bias.
- **Snapshot counts excluded:** scraped likes, views and follower counts reflect the
  scrape time, not old prediction dates. They are stored for inspection but not
  used in historical model features.
- **Explicit demo mode:** synthetic inputs make the complete workflow testable
  without paid services. Reports preserve that marker even after reimport.

External actor coverage, deleted/edited X posts, stock splits, short sample size,
lexicon quality, and omitted borrow/market-impact costs remain limitations. Live
credentials and a real-data evaluation are required for a completed empirical
submission. An attractive demo score would not establish trading performance.
