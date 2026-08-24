# Curation Notes

## Organoids_Brightfield__10 Reclassification

- **Original source group:** `by_sample_type/Organoids_Brightfield/Organoids_Brightfield__10.md`
- **Curated output location:** `new/Organoids_Fluorescence/Organoids_Brightfield__10.md`
- **Issue:** The source file was grouped under `Organoids_Brightfield`, and the user task asks for organoid time-series imaging in a 96-well plate under a 20x objective.
- **Execution evidence:** The raw planner/executor configures FITC fluorescence mode, sets fluorescence-style acquisition parameters, and performs a 24-hour hourly Z-stack time-series acquisition across the 96-well plate.
- **Curation choice:** Reclassified the curated output into `Organoids_Fluorescence` while keeping the original filename for traceability.
- **Record interpretation:** Keep the expected result aligned with the user task, and describe the execution result as FITC fluorescence acquisition.
