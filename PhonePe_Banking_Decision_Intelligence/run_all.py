"""Run the complete capstone pipeline in the required order."""
import subprocess,sys
for script in ['generate_data.py','part1.py','part2.py','part3.py','part4.py']:
    print(f'\n--- Running {script} ---',flush=True)
    subprocess.run([sys.executable,script],check=True)
print('\nAll scripts completed. Check the outputs/ directory.')
