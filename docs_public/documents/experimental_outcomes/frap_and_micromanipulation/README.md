# FRAP and Micromanipulation (Fig. 5c, Fig. 5d, Fig. S2)

FRAP + micromanipulation driving tasks and tools.

## Structure

- `frap/frap_test/` - processed FRAP test session display MDs
  (cell bleaching, letter drawing, geometric patterns, spiral drawing, 36-point circle installed/uninstalled).

- `mp285/` - processed MP-285 tool interaction records
  (movement, circle drawing, pump aspiration/dispensing, ball transfer tasks).

- VLA micromanipulation research code and ACT weight materials live under `docs_public/VLA/`.

## Note

The FRAP runtime dependency (`frap_ui_profile.json` + `record_frap_click_once.py`) is not duplicated here;
it lives at the top-level `docs_public/frap/` and is read by `tool/frap.py`.
