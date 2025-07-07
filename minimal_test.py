#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
print("Python version:", sys.version)
print("Python executable:", sys.executable)

try:
    import streamlit as st
    print("Streamlit imported successfully")
except Exception as e:
    print("Streamlit import error:", e)

try:
    from algorithms.fault_detection import FaultDetector
    print("FaultDetector imported successfully")
except Exception as e:
    print("FaultDetector import error:", e)

print("Test completed") 