import os
from src.preprocessing import preprocess
from src.porosity import calculate_porosity
from src.tortuosity import calculate_tortuosity
from src.cpk import calculate_cpk

image_files = [f for f in os.listdir(
    "data") if f.endswith(".png") or f.endswith(".jpg")]
image_files.sort()

porosity_values = []
results = []

for filename in image_files:
    path = f"data/{filename}"
    print(f"Processing {filename}...")

    img, blurred, binary = preprocess(path)
    porosity = calculate_porosity(binary)
    tortuosity, path_coords = calculate_tortuosity(binary)

    porosity_values.append(porosity)
    results.append({
        'file': filename,
        'porosity': porosity,
        'tortuosity': tortuosity
    })

    print(f"  Porosity: {porosity}%  Tortuosity: {tortuosity}")

print("\n--- Process Capability Report ---")
cpk_results = calculate_cpk(porosity_values)
print(f"Mean Porosity: {cpk_results['mean']}%")
print(f"Std Dev: {cpk_results['std']}%")
print(f"Cpk: {cpk_results['Cpk']}")
print(f"Verdict: {cpk_results['verdict']}")
