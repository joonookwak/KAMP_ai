"""Report entry point for preserved improvement/error-analysis implementation."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1]/'comparisons/improvements/code/analyze.py'),run_name='__main__')
