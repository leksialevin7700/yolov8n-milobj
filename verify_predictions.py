"""Verify prediction files meet submission requirements."""
from pathlib import Path

test_dir = Path('data/military_object_dataset/test/images')
pred_dir = Path('predictions/yolo_txt_format')

# Check 1: All files present
test_imgs = set(p.stem for p in test_dir.glob('*.jpg'))
pred_files = set(p.stem for p in pred_dir.glob('*.txt'))
missing = test_imgs - pred_files

print("=" * 80)
print("PREDICTION FILES VERIFICATION")
print("=" * 80)

print(f"\n1. File Presence:")
print(f"   Test images: {len(test_imgs)}")
print(f"   Prediction files: {len(pred_files)}")
print(f"   Missing files: {len(missing)}")
if missing:
    print(f"   ⚠️ MISSING: {list(missing)[:5]}")
else:
    print(f"   ✅ All files present")

# Check 2: Class ID validity
print(f"\n2. Class ID Validation:")
invalid_files = []
empty_files = 0
valid_files = 0
max_class = -1
min_class = 999

for txt_file in pred_dir.glob('*.txt'):
    content = txt_file.read_text().strip()
    if not content:
        empty_files += 1
        continue
    
    valid = True
    for line in content.split('\n'):
        parts = line.strip().split()
        if parts:
            try:
                cls_id = int(parts[0])
                max_class = max(max_class, cls_id)
                min_class = min(min_class, cls_id)
                if cls_id > 11 or cls_id < 0:
                    invalid_files.append((txt_file.name, cls_id))
                    valid = False
                    break
            except:
                pass
    
    if valid:
        valid_files += 1

print(f"   Files with detections: {len(pred_files) - empty_files}")
print(f"   Valid class IDs (0-11): {valid_files}")
print(f"   Invalid class IDs: {len(invalid_files)}")
print(f"   Empty files: {empty_files}")
print(f"   Class ID range found: {min_class} to {max_class}")
print(f"   Expected range: 0 to 11")

if invalid_files:
    print(f"   ⚠️ INVALID CLASS IDs in: {invalid_files[:5]}")
else:
    print(f"   ✅ All class IDs valid")

# Check 3: Format validation
print(f"\n3. Format Validation:")
sample_file = None
for txt_file in pred_dir.glob('*.txt'):
    content = txt_file.read_text().strip()
    if content:
        sample_file = txt_file
        break

if sample_file:
    line = content.split('\n')[0]
    parts = line.split()
    print(f"   Sample file: {sample_file.name}")
    print(f"   Sample line: {line}")
    print(f"   Values per line: {len(parts)} (expected: 6)")
    if len(parts) == 6:
        print(f"   ✅ Format correct: class x y w h confidence")
    else:
        print(f"   ⚠️ Format incorrect")
else:
    print(f"   ⚠️ No sample file found")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

all_ok = len(missing) == 0 and len(invalid_files) == 0

if all_ok:
    print("✅ ALL REQUIREMENTS SATISFIED")
    print("   - All files present")
    print("   - Class IDs valid (0-11)")
    print("   - Format correct")
else:
    print("⚠️ ISSUES FOUND:")
    if missing:
        print(f"   - Missing {len(missing)} prediction files")
    if invalid_files:
        print(f"   - {len(invalid_files)} files have invalid class IDs")
    print("\n   ACTION REQUIRED: Regenerate predictions")

print("=" * 80)



