#!/usr/bin/env python

import foampy
import subprocess

print("Performing deep clean of the case")

foampy.clean(remove_zero=True)

print("Cleaning processed")
subprocess.call("rm processed/log.*", shell=True)
subprocess.call("rm -r processed/fields", shell=True)
subprocess.call("rm -r processed/postProcessing.*", shell=True)

