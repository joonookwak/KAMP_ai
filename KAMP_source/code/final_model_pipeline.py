"""Report command: raw CSV -> clip splits -> GMM fitting -> per-timestamp predictions."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import argparse,subprocess,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dataset import prepare

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/final_model')
    p.add_argument('--code-dir',type=Path,default=None,help='Legacy compatibility only; uses included final source')
    p.add_argument('--window',type=int,default=3)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    args=p.parse_args();out=args.output_dir.resolve();samples,clips,provenance=prepare(args.data_dir)
    config={'seeds':args.seeds,'window':args.window,'policy':'point','source_SHA256':{Path(p['path']).name:p['sha256'] for p in provenance},'method':'StartupMarginal'}
    if (out/'pipeline_config.json').exists() and json.loads((out/'pipeline_config.json').read_text())!=config:
        raise ValueError('Output contains a different configuration; choose a new output directory')
    out.mkdir(parents=True,exist_ok=True);(out/'pipeline_config.json').write_text(json.dumps(config,indent=2)+'\n')
    data=out/'data';data.mkdir(exist_ok=True)
    for split,name in [('train','train.csv'),('validation','validation.csv'),('calibration','calibration.csv')]:samples[samples.split.eq(split)].to_csv(data/name,index=False)
    samples[samples.split.isin(['normal_test','fault_test'])].to_csv(data/'test.csv',index=False);clips.to_csv(data/'clip_splits.csv',index=False)
    subprocess.run([sys.executable,str(ROOT/'train.py'),'--data-dir',str(data),'--output-dir',str(out/'models'),'--window',str(args.window),'--seeds',*map(str,args.seeds)],check=True)
    predictions=[]
    for seed in args.seeds:
        path=out/'predictions'/f'test_predictions_seed{seed}.csv';predictions.append(str(path))
        subprocess.run([sys.executable,str(ROOT/'predict.py'),'--input',str(data/'test.csv'),'--models-dir',str(out/'models'),'--seed',str(seed),'--output',str(path)],check=True)
    subprocess.run([sys.executable,str(ROOT/'evaluate.py'),'--predictions',*predictions,'--output-dir',str(out/'results')],check=True)

if __name__=='__main__': main()
