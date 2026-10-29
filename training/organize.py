import os, re, csv, shutil

ROOT = r"/Users/hayakzan/Desktop/JUCE/NEW PROJ/data"  # your dataset root

# e.g. "noise_chipsynth_m069_440.0Hz_vel110" or "..._pw70"
PAT = re.compile(
    r'(?P<wave>noise|triangle|saw|pulse)_chipsynth_m(?P<midi>\d+)_'
    r'(?P<hz>[\d\.]+)Hz_vel(?P<vel>\d+)(?:_pw(?P<pw>\d+))?$',
    re.IGNORECASE
)

rows = []
wave_roots = ["NOISE","TRIANGLE","SAW","PULSE"]

for wave_dir in wave_roots:
    top = os.path.join(ROOT, wave_dir)
    if not os.path.isdir(top):
        continue
    for sub in os.listdir(top):
        subpath = os.path.join(top, sub)
        if not os.path.isdir(subpath):
            continue

        m = PAT.match(sub)
        if not m:
            print("WARN: skip (name not parseable):", subpath)
            continue

        # find a wav (take the first)
        wav_in = None
        for nm in os.listdir(subpath):
            if nm.lower().endswith(".wav"):
                wav_in = os.path.join(subpath, nm)
                break
        if not wav_in:
            print("WARN: no .wav in", subpath)
            continue

        # destination: move up one level with folder name
        newname = sub + ".wav"
        dst = os.path.join(top, newname)
        base, ext = os.path.splitext(dst)
        k = 1
        while os.path.exists(dst):  # avoid collisions
            dst = f"{base}_{k}{ext}"
            k += 1

        shutil.move(wav_in, dst)

        # try to remove the now-empty folder (ok if it fails)
        try:
            os.rmdir(subpath)
        except OSError:
            pass

        g = m.groupdict()
        rows.append({
            "relpath": os.path.relpath(dst, ROOT),
            "wave": g["wave"].lower(),
            "midi": int(g["midi"]),
            "freq_hz": float(g["hz"]),
            "velocity": int(g["vel"]),
            "pw_norm": (int(g["pw"])/100.0 if g.get("pw") else "")
        })

# write manifest
manifest = os.path.join(ROOT, "manifest.csv")
with open(manifest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["relpath","wave","midi","freq_hz","velocity","pw_norm"])
    w.writeheader()
    w.writerows(rows)

print(f"Done. Wrote {manifest} with {len(rows)} rows.")
