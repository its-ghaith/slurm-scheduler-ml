import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Starte Training auf: {device}")

# Erzeuge eine große Matrix für ordentlich Last
size = 15000 
x = torch.randn(size, size, device=device)

print("Benchmark läuft für ca. 30 Sekunden...")
start_time = time.time()

while time.time() - start_time < 30:
    # Intensive Matrix-Multiplikation
    y = torch.matmul(x, x)
    
print("Benchmark beendet. Energieverbrauch kann jetzt in Prometheus geprüft werden.")