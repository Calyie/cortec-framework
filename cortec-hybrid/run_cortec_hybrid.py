"""The same as `cortec-hybrid run`, for a clone without the command on PATH:

  python3 run_cortec_hybrid.py --data private.csv --synthetic synthetic.csv \\
      --schema my_schema.py --columns age,sex
  python3 run_cortec_hybrid.py --help
"""
import sys

from cortec_hybrid.run import main

if __name__ == "__main__":
    sys.exit(main(prog="run_cortec_hybrid.py"))
