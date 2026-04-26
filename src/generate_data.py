import numpy as np
import pandas as pd
from xfoil import XFoil
from xfoil.model import Airfoil

def generate_naca4(m, p, t):
    """Generate NACA 4-digit airfoil coordinates."""
    xf = XFoil()
    xf.naca(f"{m}{p}{t:02d}")
    return xf

def run_sweep(naca_code, re, aoa_start=-10, aoa_end=15, aoa_step=0.5, max_iter=100):
    """
    Run XFoil polar sweep for a given NACA 4-digit code.
    Returns a DataFrame with results, or None if it fails.
    """
    xf = XFoil()
    xf.naca(naca_code)
    xf.Re = re
    xf.max_iter = max_iter
    xf.print = False  # suppress XFoil console output

    a, cl, cd, cm, cp = xf.aseq(aoa_start, aoa_end, aoa_step)

    # Filter out non-converged points (XFoil returns NaN for those)
    mask = ~np.isnan(cl)
    if mask.sum() == 0:
        return None

    df = pd.DataFrame({
        "naca":   naca_code,
        "re":     re,
        "aoa":    a[mask],
        "cl":     cl[mask],
        "cd":     cd[mask],
        "cm":     cm[mask],
    })
    return df


def main():
    results = []

    # NACA 4-digit parameter ranges
    # m = max camber (0-9 % of chord)
    # p = position of max camber (0-9, in tenths of chord)
    # t = max thickness (6-21 % of chord)
    m_values = range(0, 5)       # 0..4
    p_values = range(0, 6)       # 0..5  (p=0 only valid when m=0)
    t_values = range(6, 22, 2)   # 6,8,10,12,14,16,18,20
    re_values = [5e5, 1e6, 2e6]  # three Reynolds numbers

    total = len(m_values) * len(p_values) * len(t_values) * len(re_values)
    count = 0

    for m in m_values:
        for p in p_values:
            # skip physically invalid combos: camber>0 but no camber position
            if m == 0 and p != 0:
                continue
            if m != 0 and p == 0:
                continue
            for t in t_values:
                naca_code = f"{m}{p}{t:02d}".zfill(4)
                for re in re_values:
                    count += 1
                    print(f"[{count}] NACA {naca_code}  Re={re:.0e}", end=" ... ")
                    df = run_sweep(naca_code, re)
                    if df is not None:
                        results.append(df)
                        print(f"OK ({len(df)} points)")
                    else:
                        print("no convergence, skipped")

    if results:
        dataset = pd.concat(results, ignore_index=True)
        dataset.to_csv("data/xfoil_dataset.csv", index=False)
        print(f"\nDone. {len(dataset)} rows saved to data/xfoil_dataset.csv")
    else:
        print("No data generated.")

if __name__ == "__main__":
    main()