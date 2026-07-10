r"""Fetch small, public real 1-D signals for the aliasing experiments.

Each signal is saved as ``data/<name>.npz`` with fields ``x`` (mono float32, native
resolution) and ``fs`` (sample rate or nominal cadence).  Sources are public / redistributable
and documented in the README.  Runs are idempotent (skips existing files).

Signals
-------
* ``speech``    -- Open Speech Repository "Harvard sentences" (public domain), 8 kHz. Broadband.
* ``sunspots``  -- SILSO daily total sunspot number (Royal Observatory of Belgium). Smooth,
                   quasi-periodic -> recoverable-side science signal.
* ``seismic``   -- ObsPy bundled example seismogram if available (broadband transient).
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TIMEOUT = 40

SOURCES = {
    "speech": "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav",
    "sunspots": "https://www.sidc.be/SILSO/INFO/sndtotcsv.php",
    # 13-month smoothed monthly total sunspot number -> a genuinely smooth, low-bandwidth
    # scientific signal for the recoverable side of the limit
    "sunspots_smooth": "https://www.sidc.be/SILSO/DATA/SN_ms_tot_V2.0.csv",
    # Mauna Loa monthly mean CO2 (NOAA GML) -> smooth trend + annual cycle (recoverable side)
    "co2": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "inr-aliasing-limits/0.1"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


def fetch_speech(path: Path) -> bool:
    try:
        import soundfile as sf
        raw = _get(SOURCES["speech"])
        x, fs = sf.read(io.BytesIO(raw))
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x[: int(2.0 * fs)]  # first ~2 s
        x = (x - x.mean()) / (np.std(x) + 1e-9)
        np.savez(path, x=x.astype(np.float32), fs=float(fs))
        print(f"  speech: {x.size} samples @ {fs} Hz")
        return True
    except Exception as e:  # pragma: no cover
        print(f"  speech FAILED: {e}")
        return False


def fetch_sunspots(path: Path) -> bool:
    try:
        raw = _get(SOURCES["sunspots"]).decode("utf-8", "replace")
        vals = []
        for line in raw.splitlines():
            parts = line.replace(";", " ").split()
            if len(parts) >= 5:
                try:
                    v = float(parts[4])
                    if v >= 0:
                        vals.append(v)
                except ValueError:
                    pass
        x = np.array(vals[-4000:], dtype=np.float32)  # last ~11 yr of daily numbers
        x = (x - x.mean()) / (np.std(x) + 1e-9)
        np.savez(path, x=x, fs=1.0)
        print(f"  sunspots: {x.size} daily samples")
        return True
    except Exception as e:  # pragma: no cover
        print(f"  sunspots FAILED: {e}")
        return False


def fetch_sunspots_smooth(path: Path) -> bool:
    try:
        raw = _get(SOURCES["sunspots_smooth"]).decode("utf-8", "replace")
        vals = []
        for line in raw.splitlines():
            parts = line.replace(";", " ").split()
            if len(parts) >= 4:
                try:
                    v = float(parts[3])
                    if v >= 0:
                        vals.append(v)
                except ValueError:
                    pass
        x = np.array(vals, dtype=np.float32)
        x = (x - x.mean()) / (np.std(x) + 1e-9)
        np.savez(path, x=x, fs=12.0)  # monthly cadence
        print(f"  sunspots_smooth: {x.size} smoothed monthly samples")
        return True
    except Exception as e:  # pragma: no cover
        print(f"  sunspots_smooth FAILED: {e}")
        return False


def fetch_co2(path: Path) -> bool:
    try:
        raw = _get(SOURCES["co2"]).decode("utf-8", "replace")
        vals = []
        for line in raw.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    v = float(parts[3])  # monthly average column
                    if v > 0:
                        vals.append(v)
                except ValueError:
                    pass
        x = np.array(vals, dtype=np.float32)
        x = (x - x.mean()) / (np.std(x) + 1e-9)
        np.savez(path, x=x, fs=12.0)
        print(f"  co2: {x.size} monthly samples")
        return True
    except Exception as e:  # pragma: no cover
        print(f"  co2 FAILED: {e}")
        return False


def fetch_seismic(path: Path) -> bool:
    try:
        from obspy import read  # type: ignore
        st = read()  # ObsPy ships a 3-component example event
        tr = st[0]
        x = np.asarray(tr.data, dtype=np.float32)
        x = (x - x.mean()) / (np.std(x) + 1e-9)
        np.savez(path, x=x, fs=float(tr.stats.sampling_rate))
        print(f"  seismic: {x.size} samples @ {tr.stats.sampling_rate} Hz")
        return True
    except Exception as e:  # pragma: no cover
        print(f"  seismic SKIPPED (obspy not available): {e}")
        return False


def main(names=None):
    fetchers = {"speech": fetch_speech, "sunspots": fetch_sunspots,
                "sunspots_smooth": fetch_sunspots_smooth, "co2": fetch_co2,
                "seismic": fetch_seismic}
    names = names or list(fetchers)
    ok = {}
    for n in names:
        p = HERE / f"{n}.npz"
        if p.exists():
            print(f"{n}: exists, skip")
            ok[n] = True
            continue
        print(f"{n}: fetching ...")
        ok[n] = fetchers[n](p)
    print("summary:", ok)
    return ok


if __name__ == "__main__":
    main(sys.argv[1:] or None)
