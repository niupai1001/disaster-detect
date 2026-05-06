# Research Sprint Brief: Remote Sensing Disaster Delineation Methods

## Objective

Run a research-and-learning sprint before formal project implementation. The goal is to learn how similar remote-sensing disaster localization and segmentation problems are handled, then produce source-grounded method options for later adversarial brainstorming.

## Scope

Study methods for:

- wildfire / active fire / burned-area / burn-severity mapping;
- landslide, debris-flow, shallow-landslide, and mass-movement mapping;
- disaster change detection and post-disaster damage assessment;
- weak supervision, sparse labels, coarse polygons, pseudo masks;
- remote-sensing and geospatial foundation models;
- detection fallbacks only as secondary localization routes.

## Required Learning Log Schema

For every paper or major source studied, write a readable Chinese Markdown learning log that includes:

- what problem the paper/source aims to solve;
- what data it uses;
- what methods it uses;
- what its innovations are;
- what the main experimental conclusions and limitations are;
- what it teaches this project;
- what constraints it adds to the later brainstorm;
- what should be checked next.

The learning log must be about the paper itself first, then about this project. It must not be only a relevance note.

## Required Outputs

- One Chinese learning-log Markdown file for every paper or major source studied.
- A method/source matrix summarizing source, task, input data, labels, output type, method family, metrics or evidence type, openness, and relevance to this project.
- A research memo that explains what the research department learned and what the later brainstorm must absorb.

## Rules

- Do not select the final model during this sprint.
- Do not start training or implementation.
- Do not treat YOLO as the default solution.
- Keep learning synchronized with the user by writing readable logs, not just private notes.
- Later departments must absorb these research outputs before the real brainstorm.
