# Data and evidence provenance

## Source panel

The project uses Google Meridian's official **simulated** geo-level example panel: 6,240 rows, 40 geos and 156 weekly observations from 2021-01-25 to 2024-01-15. Channel0–Channel4 remain anonymous; no channel labels are invented.

- [Source CSV](https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/geo_all_channels.csv)
- [Upstream repository](https://github.com/google/meridian)
- [Upstream license](https://github.com/google/meridian/blob/main/LICENSE)
- Analyzed file SHA-256: `d9ee016f7cd21f5c90b50da10a794af91a372b69f881d3caa266edd41fdafbf6`

The source CSV is not bundled. The downloader verifies this exact checksum and stops if the upstream file has changed. The URL follows an upstream branch, not an immutable archive; long-term availability of that exact version is not guaranteed. A verified local copy can be supplied to the reproduction runner. Upstream data retain their upstream terms; this project does not relicense them.

## Hypothetical experimental evidence

`data/illustrative_experiment/evidence.json` declares a return estimate of 1.20 and SE 0.15 for a specified Channel2 +20% intervention. The estimate and precision are assumed for this calibration scenario. There is no underlying real experiment or claimed achieved power for this summary.

`src/prepare_experiment_scenario.py` reconstructs the counterfactual input schedules from training-only data. It does not read or generate experimental outcomes. `src/illustrative_calibration.py` incorporates the evidence once via a likelihood, preserving original historical priors.

## Saved results

The reports are selected snapshots of previously executed runs. Runtime environments, raw posterior binaries, intermediate fits and local filesystem paths are not part of the release. Some audit metadata retain hashes from the original analysis sources and omitted posterior arrays; those are historical provenance, not claims that the packaged files have identical hashes after path-only refactoring. Reproduction writes new outputs in a separate directory.
