import subprocess
import sys

yaml_file = "environment.yml"

try:
    results = subprocess.run(
        ["conda", "env", "create", "-f", yaml_file],
        check=True,
        text=True,
        capture_output=True,
    )
    print("Environment created successfully.")
except subprocess.CalledProcessError as e:
    print("Error creating environment:")
    print(e.stderr)
    sys.exit(1)

    