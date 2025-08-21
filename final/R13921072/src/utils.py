# utils.py
import csv

def write_csv(filename, predictions, loader):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'prediction'])
        for i, pred in enumerate(predictions):
            writer.writerow([i, pred])
