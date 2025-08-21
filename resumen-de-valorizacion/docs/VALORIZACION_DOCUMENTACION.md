Version: 1.0
Generated: 2025-08-18

Overview
--------
This document describes how to run, validate and maintain `VALORIZACION-CON-PLANTILLA.py`. It covers inputs, processing rules, the important selection algorithm preserved from the legacy tool, classification rules, memory/performance tips and troubleshooting.

Contents
--------
- Purpose
- Quick start
- Input files and required columns
- Column-detection heuristics
- Processing pipeline (step-by-step)
- Core algorithms and decisions
- DIARIO / WALKBY re-classification policy
- Memory, performance and scaling guidance
- Troubleshooting and common failures
- Validation and recommended tests
- Maintenance checklist
- Appendix: outputs and regeneration

Purpose
-------
`VALORIZACION` converts raw meter-reading exports into an auditable Excel workbook suitable for downstream billing and reporting. Outputs include:

- A pivot (`meter × date`) with daily counts
- Per-meter classification (DIARIO, INTERMITENTE, WALKBY, etc.)
- `VALORIZABLE` — rows selected as final readings used for valuation
- `ELIMINADOS` — every removed reading with a removal reason

Quick start
-----------
1. Put your input files into a folder.
2. Run the script (a file picker opens):

```powershell
py -3 VALORIZACION.py
```

Or drag-and-drop files onto the script in this order: `data_file`, `meters_file` (optional), `incidents_file` (optional).

An output workbook named `dep_<input_basename>.xlsx` will be created next to the data file.

Input files
-----------
Data file (required)
- Excel `.xls`/`.xlsx` with one or more sheets (all concatenated).
- Required logical columns: meter ID, a date or combined datetime (`Record Time`), optional separate time column (`Record Time_time` or similar), and at least one numeric column (flow/counter).

Meters file (optional)
- Excel file listing meters (single column OK). If provided, meters missing from the data file are included in the pivot with `SIN LECTURA` classification.

Incidents file (optional)
- Simple two-column sheet: meter ID and incident type. Incidents can force classification or removal.

Column detection heuristics
---------------------------
The script uses case-insensitive substring matching to find common columns:

- Meter ID: headers containing `meter id`, `meter_id`, `meterid`, `medidor`, `id_medidor`
- Date/DateTime: headers containing `record time` (but avoid `time_time` which is the separate time column)
- Time: headers containing `time_time`, `time`, or `record time_time`
- Flow/volume: numeric columns containing `flow`, `count`, `total`, or `volume`. If none found, the first numeric column is chosen.

If the file has a single DateTime column, `split_datetime_column()` will split it into date and time.

Processing pipeline (high level)
------------------------------
1. Read & concatenate all sheets from the data file.
2. Read optional meters/incidents files.
3. Optimize memory (downcast numerics, convert safe objects to `category`).
4. Detect `Gateway` vs `WALKBY` rows and split data.
5. For Gateway rows: limit readings per meter/day to `MAX_READINGS_PER_DAY` (default 7) using the legacy selection function `remove_closest_readings_fast()` to guarantee identical selection behavior.
6. For WALKBY rows: keep only the latest reading per meter/day.
7. For meters that are pure WALKBY (no Gateway rows): keep only the first reading per month.
8. Record every removed row in `ELIMINADOS` with `Motivo_Eliminacion`.
9. Build pivot table (`meter × date`) counting readings.
10. Compute `Numero De Lecturas` and `Dias Lecturadas` per meter per month.
11. Apply monthly classification logic.
12. Apply incident-driven removals and final DIARIO/WALKBY conflict resolution.
13. Write the Excel workbook with the expected sheets.

Core algorithms
---------------
- remove_closest_readings_fast(timestamps, indices, max_readings)
  - Ported verbatim from the legacy `temp-valorization.py` to produce bit-for-bit identical reductions for Gateway groups. It retains first/last points and selects intermediates to maximize temporal spread.

- WALKBY rules
  - Per-day: keep the latest reading.
  - Pure-WALKBY per-month: keep the earliest reading of the month and drop the rest.

DIARIO / WALKBY re-classification
---------------------------------
The re-classification follows a deterministic two-step policy to avoid ordering bugs:

1. Identify meters tagged `DIARIO / WALKBY`.
2. For each meter, if any date has both Gateway and Walkby readings, remove that day's Walkby readings (Gateway wins) and mark the meter as `DIARIO`.
3. After all deletions, recalculate `Numero De Lecturas` and `Dias Lecturadas`.
4. Any meter still `DIARIO / WALKBY` after resolution is set to `INTERMITENTE / WALKBY`.

Memory and performance guidance
-------------------------------
- Use the included `optimize_memory_usage()` to reduce dtype sizes where safe.
- Per-group reduction runs in parallel when many groups are present; for extremely large files consider processing by month to reduce peak memory.
- If you hit `ArrayMemoryError`, either increase machine RAM or split the input.

Troubleshooting
---------------
- Missing columns (KeyError): inspect `df.columns` and confirm presence of a date/datetime column. Use `split_datetime_column()` if needed.
- Memory errors: process smaller time windows or run on larger machines.
- Unexpected classification: ensure you are running the latest script which recalculates totals after deletions.

Validation and tests
--------------------
- Unit tests to add:
  - Compare `remove_closest_readings_fast()` with the legacy function on small datasets.
  - Test `limit_readings_per_day_optimized()` for edge sizes.
  - Test pivot production and `ELIMINADOS` content under mixed data.
- Integration: prepare a 3-sheet synthetic dataset covering Gateway-only, mixed, pure-WALKBY and incidents, then run the script and compare outputs.

Maintenance checklist
---------------------
- Keep `temp-valorization.py` as golden reference for audits.
- Avoid changing `remove_closest_readings_fast()` without adding test coverage and documenting the change.
- Add `tests/` and CI (`pytest`) to guard against regressions.

Appendix: outputs & regeneration
--------------------------------
- Typical output workbook: `dep_<input>.xlsx` with sheets `TABLA DINAMICA`, `VALORIZABLE`, `ELIMINADOS`.

Regenerating the PDF from Markdown

If you update either Markdown file you can regenerate the PDF(s) using the included converter:

```powershell
py -3 docs\make_pdf.py docs\VALORIZACION_DOCUMENTACION.md docs\VALORIZACION_DOCUMENTACION.pdf
py -3 docs\make_pdf.py docs\VALORIZACION_DOCUMENTACION_ES.md docs\VALORIZACION_DOCUMENTACION_ES.pdf
```

Contact & handover
------------------
Provide the following to your team for a smooth handover:

- This documentation (English & Spanish) in `docs/`.
- A short demo run with a representative data file.
- The original `temp-valorization.py` for parity checks.

End of document

