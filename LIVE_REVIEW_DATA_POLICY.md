# Live Review Data Policy

The hiring-manager LIVE REVIEW path is designed to contain **no synthetic game, player, or market inputs**.

It uses:

- current NBA scoreboard / box score / play-by-play from the live NBA feed;
- current public Kalshi market data via REST;
- a current Dunks & Threes snapshot generated privately by GitHub Actions from the repository owner's credential.

The D&T credential is never committed to the repository and is never made available to the reviewer. The reviewer retrieves only the current snapshot from the private `review-data` branch.

## Important licensing condition

Before enabling the scheduled refresher for an external reviewer, the repository owner must confirm that their Dunks & Threes subscription / terms permit sharing the captured or derived data with that reviewer. The code intentionally does not bypass provider authentication.

If sharing is not permitted, leave the refresher disabled and do not distribute `review_data/current.json`.

## Trading safety

`scripts/run_live_review.sh` is PAPER-only. It does not pass `--live`, `--pilot-live`, or `--ws`.

A reviewer can therefore evaluate:

- current-game probability generation;
- current public market comparison;
- candidate generation;
- portfolio sizing;
- fresh-state / fresh-book checks;
- paper maker-order lifecycle;

without an exchange account and without the ability to submit real-money orders.
