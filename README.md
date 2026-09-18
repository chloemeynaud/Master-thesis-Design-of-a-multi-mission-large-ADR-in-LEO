# Master thesis — Design of a multi-mission large ADR in LEO

This repository contains all the Python scripts developed as part of my master
thesis. The thesis is separated into three phases: client choice, mission design
and preliminary system design. Code was developed for each of these phases and is
contained in the `scripts` folder. The `data` folder contains the Excel files the
scripts read.

## Repository structure

```
.
├── scripts/     runners and analysis modules
├── data/        input workbooks (catalogue and target list)
└── outputs/     figures and tables produced by the runners (not versioned)
```

## Running the code

Three scripts are meant to be run directly. Each one produces the figures and
tables of one phase of the thesis:

| Script | Phase |
|---|---|
| `run_client_selection.py` | Phase 1 — Client choice |
| `run_mission_design.py` | Phase 2 — Mission design |
| `run_system_design.py` | Phase 3 — Preliminary system design |

Every runner contains only the sequence of calls, in the same order as the
corresponding chapter of the thesis, so that a figure in the report can be traced
back to the lines that produced it. Each one starts with a `STEPS` dictionary:
setting an entry to `False` skips that step, which is convenient when only one
figure needs to be regenerated. Setting `SHOW = False` writes the PNG files
without opening the figure windows.

All outputs are written to `outputs/`, which is recreated on each run and is not
versioned.

## The other files

The remaining files in `scripts/` are modules: they hold the functions the
runners call and are not executed on their own.

## Input data

| File | Contents |
|---|---|
| `RB_Payloads_notdecay_LEO_with_DISCOS.xlsx` | The object catalogue: every rocket body and payload still in orbit in LEO, with the orbital elements from Space-Track and the physical properties from DISCOS. One sheet per object type. |
| `Targets.xlsx` | The objects selected for removal in each cluster, one sheet per cluster, with their RAAN. |

## Requirements

Python 3.9 or later, with:

```bash
pip install pandas numpy matplotlib openpyxl requests
```
```


