import sys
import inspect
import trl
from trl import PPOConfig

print("--- PYTHON ENVIRONMENT DIAGNOSTIC ---")
print(f"Python Executable Being Used: {sys.executable}")
print("-" * 30)

try:
    print(f"Reported TRL Version: {trl.__version__}")
    print(f"Location of 'trl' module: {trl.__file__}")
    print(f"Location of 'PPOConfig' class: {inspect.getfile(PPOConfig)}")
except Exception as e:
    print(f"An error occurred during inspection: {e}")

print("-" * 30)
print("Python's Search Path (sys.path):")
# We print this in reverse to see what Python prioritizes first
for path in reversed(sys.path):
    print(f"  - {path}")
print("--- END OF DIAGNOSTIC ---")