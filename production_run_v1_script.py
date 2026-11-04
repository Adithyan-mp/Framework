import subprocess

machines = [
    '172.16.64.39',
    '172.16.64.36',
    '172.16.64.37',
    '172.16.64.38',
    '172.16.64.40',
]

processes = []

for i, ip in enumerate(machines):

    config = f"v1_unit_test/config/config_{i}.yamal"

    

    print(f"Launching rank {i} on {ip}")

    if i == 0:
        command = f"python3 -m v1_unit_test.main {config}"
        process = subprocess.Popen(
            ['bash', '-lc', command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    else:
        command = f"cd adi/Framework && source venv/bin/activate && python3 -m v1_unit_test.main {config}"

        process = subprocess.Popen(
            ['ssh', ip, 'bash', '-lc', f"'{command}'"], 
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

    print("STDOUT:")
    print(stdout)

    print("STDERR:")
    print(stderr)

    print("Return code:", process.returncode)

    print("=" * 50)