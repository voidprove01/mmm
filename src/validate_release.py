"""Fast, offline integrity checks for the public portfolio (standard library)."""
from pathlib import Path
import ast,json,re
ROOT=Path(__file__).resolve().parents[1]
notebooks=sorted((ROOT/'notebooks').glob('*.ipynb'))
assert len(notebooks)==3
for path in notebooks:
 n=json.loads(path.read_text())
 for cell in n['cells']:
  if cell['cell_type']=='code':
   assert cell.get('execution_count') is not None,path
   assert not any(o['output_type']=='error' for o in cell.get('outputs',[])),path
for path in [ROOT/'reproduce.py',*(ROOT/'src').glob('*.py')]:ast.parse(path.read_text())
for path in ROOT.rglob('*.md'):
 if 'runs' in path.relative_to(ROOT).parts:continue
 for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
  if target.startswith(('http:','https:','#','mailto:')):continue
  assert (path.parent/target.split('#')[0]).exists(),(path,target)
for name in ['mcmc','illustrative_calibration']:
 d=json.loads((ROOT/'reports'/name/'diagnostics.json').read_text())
 assert d['divergences']==0 and d['max_rhat']<1.01
c=json.loads((ROOT/'reports/illustrative_calibration/comparison_summary.json').read_text())
assert c['experiment']['observed_incremental_return']==1.2
assert c['experiment']['standard_error_incremental_return']==.15
assert c['experiment_return_after']['sd']<c['experiment_return_before']['sd']
for name in ['budget_optimization','calibrated_budget']:
 m=json.loads((ROOT/'reports'/name/'summary.json').read_text())
 assert m['bounds']==[.8,1.2]
 assert m['revenue_gain']['lo90']<0<m['revenue_gain']['hi90']
e=json.loads((ROOT/'reports/experiment_prioritization/summary.json').read_text())
assert e['draws']==4000 and e['no_experiment_evsi_cu']==0
import csv
with (ROOT/'reports/experiment_prioritization/candidate_results.csv').open() as handle:rows=list(csv.DictReader(handle))
assert len(rows)==15
assert all(0<=float(r['evsi_cu'])<=e['evpi_cu'] for r in rows)
assert all(abs(float(r['predictive_mass'])-1)<2e-5 for r in rows)
with (ROOT/'reports/channel0_designs/design_comparison.csv').open() as handle:designs=list(csv.DictReader(handle))
assert len(designs)==9
assert all(0<=float(r['evsi_cu'])<=e['evpi_cu'] for r in designs)
for tier in ['low','medium','high']:
 group=[r for r in designs if r['precision']==tier]
 assert len(group)==3 and len({r['se_revenue_cu'] for r in group})==1
 assert all(float(r['spend_change_cu'])*float(r['revenue_change_mean_cu'])>0 for r in group)
for path in ROOT.rglob('*'):
 if not path.is_file() or 'runs' in path.relative_to(ROOT).parts:continue
 assert path.suffix not in {'.nc','.npz'},path
 if path.suffix in {'.py','.md','.json','.ipynb','.csv','.txt'}:
  assert str(Path.home()) + '/' not in path.read_text(errors='replace'),path
print('PASS: three executed notebooks, valid source syntax and links, coherent result snapshots, no large posterior binaries or local home paths.')
