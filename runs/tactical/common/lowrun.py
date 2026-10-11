"""usage: python lowrun.py script.py [args...]  - runs the script at BELOW_NORMAL priority (recut jobs share the machine)."""
import ctypes, os, runpy, sys
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass
script = os.path.abspath(sys.argv[1]); sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
