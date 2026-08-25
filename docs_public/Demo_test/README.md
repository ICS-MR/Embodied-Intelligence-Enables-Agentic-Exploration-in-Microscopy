# Demo Test

Interactive end-to-end demo sessions of the EIMS workflow. Each subdirectory is a self-contained execution showcase with a display Markdown and the generated `output/` artifacts.

## Contents

| Directory | Rounds | Output artifacts |
| --- | ---: | --- |
| `bright_spot_detection_and_60x_acquisition/` | 4 | Brightfield overview, bright-spot detection JSON, and 11 60x bright-spot field images. |
| `multichannel_fluorescence_merge/` | 1 | Multi-channel fluorescence stack (DAPI/FITC/TRITC) and merged RGB composite. |

## Showcase format

Each showcase Markdown follows the standard EIMS execution-record layout:

- `# EIMS Workflow Showcase: <Title> (Execution Record)`
- `## Round N` per user interaction round
  - `### User Input (Natural Language)` - the natural-language instruction
  - `### Planner-Generated Plan` - ordered subtask list with assigned module
  - `### Executor-Generated Restricted Python Code (Complete)` - generated code grouped by module

The `output/` folder holds the image and detection artifacts produced during the session.
