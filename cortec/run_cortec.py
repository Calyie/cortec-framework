"""The same as `cortec run`, for a clone without the command on PATH:

  python3 run_cortec.py --data private.csv --schema my_schema.py --backend mock
  python3 run_cortec.py --help
"""
import sys

from cortec.run import main

if __name__ == "__main__":
    sys.exit(main(prog="run_cortec.py"))
