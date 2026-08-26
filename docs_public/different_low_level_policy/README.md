# Different Low-Level Policy Baselines

This folder contains standalone baseline implementations used to compare different
low-level execution policies against the EIMS agentic workflow. These materials are kept
under `docs_public/` for reproducibility and public review, but they are separate from the
core EIMS runtime.

The VLM and VLA execution pipelines are not intrinsic components of the EIMS agent; rather, they were implemented as standalone code modules to serve as comparative baselines for different low-level execution policies.

## Purpose

EIMS uses a planner/executor/checker architecture that converts natural-language
experimental intent into confirmed, tool-mediated microscope workflows. The baselines in
this folder test alternative low-level execution strategies, including direct VLM-based
visual localization or ACT/VLA-style policy inference, so their behavior can be compared
with the EIMS workflow in manuscript analyses.

## Directory Structure

| Path | Content |
| --- | --- |
| `VLM/vlm_location_comparison/` | VLM localization comparison workflow against local MMDetection predictions. |
| `VLM/vlm_focus_and_brightness/` | VLM focus and brightness benchmark scripts. |
| `ACT_VLA/Micromanipulation_tool/` | ACT-style VLA micromanipulation code for data collection, training, and inference. |
| `ACT_VLA/ACT_for_microscopy/` | Placeholder and layout notes for the external ACT/VLA weight bundle. |

## Scope Notes

- These baselines are not required for the default EIMS runtime, Micro-Manager demo mode, real microscope setup, Fiji/ImageJ integration, or the standard planner/executor/checker workflow.
- VLM materials focus on visual perception and low-level decision comparisons, not on replacing the EIMS planner or execution runtime.
- ACT/VLA materials provide policy-training and inference code for comparative low-level control experiments; model weights and large external assets should be restored separately as documented in the corresponding subfolder.
- Runtime configuration for EIMS remains governed by the root README and `config/runtime_config.example.json`, not by the baseline folders here.
