"""Report entry point; baseline implementations are grouped in comparisons/baselines."""
from pathlib import Path
import sys,runpy,importlib.util
_ROOT=Path(__file__).resolve().parents[1]
_IMPL=_ROOT/'comparisons/baselines/run_experiment.py'
sys.path.insert(0,str(_IMPL.parent))
if __name__=='__main__':
    runpy.run_path(str(_IMPL),run_name='__main__')
else:
    _spec=importlib.util.spec_from_file_location('_baseline_run_experiment',_IMPL)
    _module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
    globals().update({k:v for k,v in vars(_module).items() if not k.startswith('__')})
