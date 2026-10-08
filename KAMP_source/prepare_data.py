"""Optional: recreate the included train/calibration/test CSVs from raw data."""
import argparse, json
from pathlib import Path
from src.dataset import prepare
ROOT = Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, default=ROOT/'data/raw')
    p.add_argument('--output-dir', type=Path, default=ROOT/'data')
    args=p.parse_args()
    samples, clips, provenance=prepare(args.raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names={'train':'train.csv','validation':'validation.csv','calibration':'calibration.csv'}
    for split, filename in names.items():
        samples.loc[samples.split.eq(split)].to_csv(args.output_dir/filename,index=False)
    samples.loc[samples.split.isin(['normal_test','fault_test'])].to_csv(args.output_dir/'test.csv',index=False)
    clips.to_csv(args.output_dir/'clip_splits.csv',index=False)
    for entry in provenance: entry['path']=Path(entry['path']).name
    (args.output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(clips.groupby('split').size().to_string())

if __name__=='__main__': main()
