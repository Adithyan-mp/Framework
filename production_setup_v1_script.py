
import subprocess

machines = [
    '172.16.64.36',
    '172.16.64.37',
    '172.16.64.38',
    '172.16.64.40',
    '172.16.64.41'
]

for ip in machines:
    print(f"Machine {ip} started initializing")

    commands = (
        "mkdir -p adi && "
        "cd adi && "
        "git clone -b production https://github.com/Adithyan-mp/Framework.git && "
        "cd Framework && "
        "python3 -m venv venv && "
        "./venv/bin/pip install -r requirements.txt"
    )

    res = subprocess.run(
        ['ssh', ip, 'bash', '-lc', commands],
        capture_output=True,
        text=True
    )

    if res.returncode == 0:
        print(f"Successfully initialized machine {ip}")
        print(res.stdout)
    else:
        print(f"Failed to initialize machine {ip}")
        print(res.stderr)
        raise RuntimeError(f"Initialization failed on {ip}")

    print("=" * 50)
