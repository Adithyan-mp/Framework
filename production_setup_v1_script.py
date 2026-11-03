import subprocess

machines = [
    '172.16.64.36',
    '172.16.64.37',
    '172.16.64.38',
    '172.16.64.40',
    '172.16.64.41'
]

successful = []
failed = []

commands = """
mkdir -p adi &&
cd adi &&
git clone -b production https://github.com/Adithyan-mp/Framework.git &&
cd Framework &&
python3 -m venv venv &&
./venv/bin/pip install -r requirements.txt
"""

for ip in machines:
    print(f"Machine {ip} started initializing", flush=True)

    try:
        res = subprocess.run(
            ['ssh', ip, 'bash', '-lc', commands],
            timeout=1800
        )

        if res.returncode == 0:
            print(f"Successfully initialized machine {ip}")
            successful.append(ip)
        else:
            print(f"Failed to initialize machine {ip}")
            failed.append(ip)

    except subprocess.TimeoutExpired:
        print(f"Machine {ip} initialization timed out")
        failed.append(ip)

    print("=" * 50)


print("\nINITIALIZATION SUMMARY")
print(f"Successful machines: {successful}")
print(f"Failed machines: {failed}")