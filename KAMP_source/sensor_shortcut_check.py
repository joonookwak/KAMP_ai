"""Report reproduction of sensor/recording sensitivity, baseline and startup GMM."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--code-dir',type=Path,default=ROOT/'code',help='Compatibility argument')
p.add_argument('--data-dir',type=Path,default=ROOT/'data')
p.add_argument('--output-dir',type=Path,default=ROOT/'runs/sensor_shortcut')
p.add_argument('--study',choices=['sensors','recording','windows','all'],default='all')
a=p.parse_args()
subprocess.run([sys.executable,str(ROOT/'comparisons/report_checks.py'),'--data-dir',str(a.data_dir),'--output-dir',str(a.output_dir),'--study',a.study],check=True)

if a.study in ['recording','all']:
    subprocess.run([sys.executable,str(ROOT/'comparisons/recording_baselines.py'),'--data-dir',str(a.data_dir),'--output-dir',str(a.output_dir/'other_methods')],check=True)
