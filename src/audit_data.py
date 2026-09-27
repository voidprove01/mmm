"""Reproducible structural audit; no third-party dependencies."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'data' / 'geo_all_channels.csv'
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    keys = Counter((row['geo'], row['time']) for row in rows)
    geos = sorted({r['geo'] for r in rows})
    dates = sorted({r['time'] for r in rows})
    columns = [c for c in rows[0] if c not in ('', 'geo', 'time')]
    missing = {c: sum(r[c] == '' for r in rows) for c in rows[0]}
    numeric = {}
    invalid = 0
    for col in columns:
        vals = []
        for row in rows:
            try:
                value = float(row[col])
                if not math.isfinite(value):
                    raise ValueError('Nonfinite')
                vals.append(value)
            except ValueError:
                invalid += 1
        numeric[col] = {'min': min(vals) if vals else None,
                        'max': max(vals) if vals else None,
                        'zero_count': sum(v == 0 for v in vals),
                        'negative_count': sum(v < 0 for v in vals)}
    weekly = all((date.fromisoformat(b) - date.fromisoformat(a)).days == 7
                 for a, b in zip(dates, dates[1:]))
    nonnegative = [c for c in columns if c.endswith(('_impression', '_spend'))]
    nonnegative += ['conversions', 'revenue_per_conversion', 'population']
    checks = {
        'unique_geo_week': max(keys.values()) == 1,
        'complete_geo_week_panel': len(keys) == len(geos) * len(dates),
        'regular_weekly_dates': weekly,
        'no_missing_cells': sum(missing.values()) == 0,
        'finite_numeric_values': invalid == 0,
        'nonnegative_media_kpi_population': all(numeric[c]['negative_count'] == 0 for c in nonnegative),
    }
    result = {'source_type': 'official simulated example, not real advertiser data',
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'rows': len(rows), 'geos': len(geos), 'weeks': len(dates),
              'first_week': dates[0], 'last_week': dates[-1],
              'checks': checks, 'missing_by_column': missing, 'numeric_summary': numeric,
              'interpretation': 'Structural checks only; no causal identification or model validation implied.'}
    out = ROOT / 'reports' / 'data_audit.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('rows', 'geos', 'weeks', 'checks')}, indent=2))
    if not all(checks.values()):
        raise SystemExit('Review failed checks before modeling.')


if __name__ == '__main__':
    main()
