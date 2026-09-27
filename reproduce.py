"""Rebuild in a new run directory; never overwrite the published snapshots."""
import argparse,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run-name',default='reproduction');p.add_argument('--local-data',type=Path);p.add_argument('--dry-run',action='store_true');args=p.parse_args()
if not args.run_name or Path(args.run_name).name!=args.run_name or args.run_name in {'.','..'}:p.error('--run-name must be a single directory name')
run=ROOT/'runs'/args.run_name
commands=[['src/download_data.py'],['src/audit_data.py'],['src/prior_predictive.py'],['src/fit_mcmc.py','--target','0.99','--dense-hyper'],['src/posterior_report.py'],['src/holdout_prediction.py'],['src/channel_results.py'],['src/optimize_budget.py'],['src/prepare_experiment_scenario.py'],['src/illustrative_calibration.py'],['src/channel_results.py','--posterior-report','illustrative_calibration','--output-report','calibrated_channels'],['src/optimize_budget.py','--posterior-report','illustrative_calibration','--channel-report','calibrated_channels','--output-report','calibrated_budget','--calibrated'],['src/report_illustrative_calibration.py'],['src/prioritize_experiments.py'],['src/expand_channel0_designs.py'],['src/execute_walkthroughs.py']]
if args.local_data:commands[0]+=['--local-file',str(args.local_data.resolve())]
if args.dry_run:
 print('Fresh output directory:',run)
 for cmd in commands:print(sys.executable,*cmd)
 raise SystemExit(0)
if run.exists():raise SystemExit('Run directory already exists. Choose a new --run-name to preserve previous results.')
run.mkdir(parents=True)
for folder in ['src','notebooks','docs']:shutil.copytree(ROOT/folder,run/folder)
shutil.copy2(ROOT/'HOLDOUT_PROTOCOL.md',run/'HOLDOUT_PROTOCOL.md')
for cmd in commands:subprocess.run([sys.executable,*cmd],cwd=run,check=True)
print('Reproduction complete:',run)
