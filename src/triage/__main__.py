"""Lets you run the pipeline as `python -m triage ...` (same as the `triage` command)."""

import sys

from triage.cli import main

sys.exit(main())
