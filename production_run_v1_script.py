
import subprocess

machines = [
    '172.16.64.39',
    '172.16.64.36',
    '172.16.64.37',
    '172.16.64.38',
    '172.16.64.40',
    '172.16.64.41'
]

processes = []

for i, ip in enumerate(machines):

    command = f"""cd /adi/Framework && 
        source venv/bin/activate && 
        python3 -m v1_unit_test.main 
        f"v1_unit_test/config/config_{i}.yamal"""

    print(f"Launching rank {i} on {ip}")

    if i == 0:
        process = subprocess.Popen(
            ['bash', '-lc', command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    else:
        process = subprocess.Popen(
            ['ssh', ip, 'bash', '-lc', command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    processes.append((i, ip, process))


# Collect results
for rank, ip, process in processes:

    stdout, stderr = process.communicate()

    print(f"\nRank {rank} | {ip}")
    print("=" * 50)

    print(stdout)

    if process.returncode != 0:
        print("ERROR:", stderr)

    print("=" * 50)
