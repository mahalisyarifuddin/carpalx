
import subprocess
import re

def run_python():
    cmd = ["python3", "carpalx.py", "-conf", "etc/carpalx.conf", "-corpus", "sample_corpus.txt", "-action", "loadkeyboard,loadtriads,reporteffort,quit"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def run_perl():
    cmd = ["perl", "-I", "legacy/lib", "legacy/bin/carpalx", "-conf", "etc/carpalx.conf", "-corpus", "/app/sample_corpus.txt", "-action", "loadkeyboard,loadtriads,reporteffort,quit"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

python_out = run_python()
perl_out = run_perl()

def extract_effort(out):
    match = re.search(r"all\s+([\d\.]+)", out)
    if match:
        return float(match.group(1))
    # Python might output it differently
    match = re.search(r"Total Effort:\s+([\d\.]+)", out)
    if match:
        return round(float(match.group(1)), 3)
    return None

py_effort = extract_effort(python_out)
pl_effort = extract_effort(perl_out)

print(f"Python Effort: {py_effort}")
print(f"Perl Effort: {pl_effort}")

if py_effort == pl_effort:
    print("MATCH!")
else:
    print("MISMATCH!")
