#!/usr/bin/env python
"""CLI entry point — delegates to main module."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from main import main

    main()
