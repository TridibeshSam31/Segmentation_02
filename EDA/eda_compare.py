import json
data = json.load(open('eda_results.json'))

print('CLASS DISTRIBUTION COMPARISON (pct of pixels):')
header = f"{'Class':20s} {'Train':>8s} {'Val':>8s} {'Test':>8s} {'T-V Delta':>10s}"
print(header)
print('-' * len(header))
classes = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky']
for cls in classes:
    t = data['train']['class_pixel_pct'][cls]
    v = data['val']['class_pixel_pct'][cls]
    te = data['test']['class_pixel_pct'][cls]
    delta = abs(t - v)
    print(f"{cls:20s} {t:8.3f} {v:8.3f} {te:8.3f} {delta:10.3f}")

print()
print('CLASS PRESENCE COMPARISON (pct of images):')
print(f"{'Class':20s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
print('-' * 50)
for cls in classes:
    t = data['train']['class_presence_pct'][cls]
    v = data['val']['class_presence_pct'][cls]
    te = data['test']['class_presence_pct'][cls]
    print(f"{cls:20s} {t:8.1f} {v:8.1f} {te:8.1f}")

print()
print('RGB STATS COMPARISON:')
for split in ['train', 'val', 'test']:
    r = data[split]['rgb_stats']
    print(f"  {split:5s} mean={r['mean']}  std={r['std']}")

print()
print('TEST unique mask values:', data['test']['all_unique_mask_values'])
print('TEST unexpected values:', data['test']['unexpected_mask_values'])

# Compute class weights (inverse frequency)
print()
print('SUGGESTED CLASS WEIGHTS (inverse freq, normalized):')
train_pct = [data['train']['class_pixel_pct'][c] for c in classes]
total = sum(train_pct)
inv_freq = [total / max(p, 0.001) for p in train_pct]
norm = min(inv_freq)
weights = [w / norm for w in inv_freq]
for c, w in zip(classes, weights):
    print(f"  {c:20s}: {w:.2f}")
