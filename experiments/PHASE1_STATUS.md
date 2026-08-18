# Phase 1 status — Tmall A0 vs A1

Live experiment state. `seeds.json` holds the locked design and does not change;
this file tracks what has actually run. Last updated 2026-08-18.

## Where things stand

B1 is implemented, tested (29 CPU tests) and committed. RetailRocket served as
the smoke test. Phase 1 on Tmall is **in progress**: run 1 of 6.

| run | arm | seed | state |
|-----|-----|------|-------|
| 1 | A0 | 2020 | running — epochs 0-1 done, resuming for 2-4 |
| 2-3 | A0 | 2021, 2022 | not started |
| 4-6 | A1 | 2020, 2021, 2022 | not started |

## Measured, not assumed

Per-epoch cost on Tmall, from run 1: **2.004 h/epoch** (7,234 batches at ~1.02
it/s, plus ~1.6 min eval).

Validation curve so far, against MBHT's published Taobao figure of 0.283:

| epoch | ndcg@10 | delta | grad steps | vs RetailRocket's whole run | % of published |
|-------|---------|-------|-----------|------------------------------|----------------|
| 0 | 0.1280 | — | 7,234 | 0.58x | 45% |
| 1 | 0.2394 | +0.1114 | 14,468 | 1.16x | 85% |

The gap to published shrank by a factor of 0.28 in one epoch. Extrapolating that
decay puts epoch 2 near 0.271, epoch 3 near 0.280 and epoch 4 near 0.282 — i.e.
roughly 99% of published by epoch 3-4, which is why the cap is 5.

**Epoch 2 is the checkpoint on that extrapolation:**

| epoch 2 lands | reading | action |
|---------------|---------|--------|
| 0.265-0.275 | extrapolation holds | continue as planned |
| below 0.255 | slower than modelled | stop and re-cost before spending more |
| above 0.280 | already at published | consider cap 4 and save ~10 units |

## Budget

Colab reported **74.21 units left at 1.07 units/hour** — 69.4 GPU hours.

| item | hours | units |
|------|-------|-------|
| finish run 1 (epochs 2-4) | 6.2 | 6.6 |
| runs 2-6, 5 epochs each | 50.1 | 53.6 |
| **total** | **56.3** | **60.2** |
| slack | | ~14 |

The slack is what funds the A1-rand control in `seeds.json`, if phase 1 shows a
gain worth attributing. `SAFETY_CAP=15` would have cost 180 hours for the phase,
two and a half times the budget.

## Next steps, in order

1. Wait for epoch 2 to finish and check it against the table above.
2. Restart Colab and resume — the running notebook still holds `cap=15` in
   memory, so it must be reloaded to pick up `cap=5`. Resume continues from the
   epoch-2 checkpoint on Drive.
3. Run 1 finishes at epoch 4 and writes its `run_meta` JSON.
4. Run the remaining five (notebook cell 9); completed runs are skipped.
5. Analysis cell reports mean+-std, Tmall's own noise floor, and Welch's t-test.

## Carried forward

- A0 is not bit-reproducible even at a fixed seed: `build_Gs_unique` assigns
  through duplicate indices (`mbht.py:503`) and breaks topk ties, both
  nondeterministic on CUDA, both inside the hypergraph the plan forbids touching.
  Measured on RetailRocket at ndcg@10 +-0.0010. Conclusions from one or two runs
  are not trustworthy; this is why the phase is multi-seed.
- `gating_bias` was uninitialised in the parent code and is now fixed
  (commit `3543355`), so seeds control everything they should.
- RetailRocket's test split contains no purchase events at all, so `CART->BUY`
  is 6.68% of its training data and 0% of its test data. It can smoke-test the
  pipeline but cannot decide whether B1 works. Tmall carries that decision.
