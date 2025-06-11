import json
import subprocess
import sys
import os
import platform

def setup_environment(config_file="config/settings.json"):
    """Sets up the project environment based on a configuration JSON file."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", config_file)
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found at {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in {config_path}")
        sys.exit(1)

    os.chdir(os.path.dirname(config_path))  # Move to root project dir

    # --- Create and activate the virtual environment ---
    venv_path = os.path.join(".venv")
    python_bin = os.path.join(venv_path, "Scripts", "python.exe") if platform.system() == "Windows" else os.path.join(venv_path, "bin", "python")

    if not os.path.exists(python_bin):
        print("🔧 Creating virtual environment...")
        subprocess.run(["python", "-m", "venv", ".venv"], check=True)

    if not os.path.exists(python_bin):
        print("❌ Failed to locate virtual environment.")
        sys.exit(1)

    print("✅ Virtual environment created or already exists.")

    # --- Execute setup commands ---
    for cmd in config.get("setup_commands", []):
        command = cmd.get("command")
        description = cmd.get("description", "")
        if not command:
            continue

        print(f"\n▶ Executing: {description}")
        print(f"$ {command}")
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running command: {command}")
            print(e.stderr)
            sys.exit(1)

    print("🎉 Environment setup complete.")

if __name__ == "__main__":
    setup_environment()
