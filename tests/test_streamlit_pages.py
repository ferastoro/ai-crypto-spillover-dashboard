"""
Automated UI / Integration test verifying all Streamlit dashboard pages via AppTest.
Ensures zero exceptions occur on page load across all application views.
"""

import os
import sys
from pathlib import Path
import pytest

BASE_DIR = str(Path(__file__).resolve().parent.parent)
os.environ["PYTHONPATH"] = BASE_DIR
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from streamlit.testing.v1 import AppTest

PAGE_FILES = [
    "app.py",
    "pages/1_Spillover.py",
    "pages/2_Forecast.py",
    "pages/3_Model_Comparison.py",
    "pages/4_Limitations.py",
    "pages/5_Latest_Forecast.py",
]


@pytest.mark.parametrize("page_rel_path", PAGE_FILES)
def test_streamlit_page_no_exception(page_rel_path: str):
    page_path = str(Path(BASE_DIR) / page_rel_path)
    at = AppTest.from_file(page_path, default_timeout=30)
    at.run()
    
    if at.exception:
        exc = at.exception[0]
        pytest.fail(f"Exception on {page_rel_path}: {exc.message}\nTraceback: {exc.value}")
    
    print(f"\n[PASSED] {page_rel_path} rendered cleanly with 0 exceptions.")


if __name__ == "__main__":
    failed = []
    for pf in PAGE_FILES:
        page_path = str(Path(BASE_DIR) / pf)
        print(f"Testing {pf}...")
        at = AppTest.from_file(page_path, default_timeout=30)
        at.run()
        if at.exception:
            exc = at.exception[0]
            print(f"  ❌ FAILED {pf}: {exc.message}")
            failed.append((pf, exc.message))
        else:
            print(f"  ✅ SUCCESS: {pf} loaded with 0 exceptions.")
    
    if failed:
        sys.exit(1)
    print("\nAll Streamlit pages verified cleanly via AppTest!")
