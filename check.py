import sys
import os

if os.environ.get('VIRTUAL_ENV'):
    print("Running in a virtual environment")
else:
    print("Running in the global Python environment")

print(f"Base prefix: {sys.base_prefix}")
print(f"Prefix: {sys.prefix}")