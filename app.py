import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="NACA Airfoil Surrogate", layout="wide")

# ── Train model ───────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv('data/xfoil_dataset.csv', dtype={'naca': str})
    df['naca'] = df['naca'].str.zfill(4)
    df['m'] = df['naca'].str[0].astype(int)
    df['p'] = df['naca'].str[1].astype(int)
    df['t'] = df['naca'].str[2:].astype(int)
    X = df[['m', 'p', 't', 're', 'aoa']]
    y = df[['cl', 'cd']]
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf

@st.cache_data
def load_data():
    df = pd.read_csv('data/xfoil_dataset.csv', dtype={'naca': str})
    df['naca'] = df['naca'].str.zfill(4)
    return df

model = train_model()
df_ref = load_data()

# ── NACA geometry ─────────────────────────────────────────────────────────────
def naca4_coords(m, p, t, n=150):
    x = np.linspace(0, 1, n)
    yt = 5*(t/100)*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                    + 0.2843*x**3 - 0.1015*x**4)
    if m == 0 or p == 0:
        yc  = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc = np.where(x < p/10,
                      (m/100)/(p/10)**2*(2*(p/10)*x - x**2),
                      (m/100)/(1-p/10)**2*((1-2*p/10)+2*(p/10)*x - x**2))
        dyc = np.where(x < p/10,
                       2*(m/100)/(p/10)**2*((p/10)-x),
                       2*(m/100)/(1-p/10)**2*((p/10)-x))
    theta = np.arctan(dyc)
    xu = x  - yt*np.sin(theta); yu = yc + yt*np.cos(theta)
    xl = x  + yt*np.sin(theta); yl = yc - yt*np.cos(theta)
    return xu, yu, xl, yl, x, yc

def make_airfoil_svg(m, p, t, width=220, height=110):
    xu, yu, xl, yl, xc, yc = naca4_coords(m, p, t)
    pad_x, pad_y = 10, 8
    w = width - 2*pad_x; h = height - 2*pad_y
    y_range = 0.28
    def tx(x): return pad_x + x * w
    def ty(y): return height/2 - y/y_range * (h/2)
    upper = " ".join(f"{tx(xu[i]):.1f},{ty(yu[i]):.1f}" for i in range(len(xu)))
    lower = " ".join(f"{tx(xl[i]):.1f},{ty(yl[i]):.1f}" for i in range(len(xl)-1, -1, -1))
    fill_path = f"M {upper} L {lower} Z"
    upper_path = "M " + " L ".join(f"{tx(xu[i]):.1f},{ty(yu[i]):.1f}" for i in range(len(xu)))
    lower_path = "M " + " L ".join(f"{tx(xl[i]):.1f},{ty(yl[i]):.1f}" for i in range(len(xl)))
    camber_path = ""
    if m > 0:
        camber_path = f'<polyline points="{" ".join(f"{tx(xc[i]):.1f},{ty(yc[i]):.1f}" for i in range(len(xc)))}" fill="none" stroke="#E24B4A" stroke-width="1" stroke-dasharray="3,3"/>'
    chord_y = ty(0)
    chord_line = f'<line x1="{tx(0)}" y1="{chord_y}" x2="{tx(1)}" y2="{chord_y}" stroke="#888" stroke-width="0.5" stroke-dasharray="3,3"/>'
    return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <path d="{fill_path}" fill="rgba(55,138,221,0.15)" stroke="none"/>
  {chord_line}{camber_path}
  <path d="{upper_path}" fill="none" stroke="#378ADD" stroke-width="1.5"/>
  <path d="{lower_path}" fill="none" stroke="#378ADD" stroke-width="1.5"/>
</svg>"""

# ── Parse NACA code ───────────────────────────────────────────────────────────
def parse_naca(code):
    code = code.strip().zfill(4)
    if len(code) != 4 or not code.isdigit():
        return None, None, None, "Invalid NACA code — must be 4 digits (e.g. 2412)"
    m, p, t = int(code[0]), int(code[1]), int(code[2:])
    if t < 4 or t > 30:
        return None, None, None, "Thickness must be between 04 and 30"
    if m == 0 and p != 0:
        return None, None, None, "Camber position must be 0 for symmetric airfoils (m=0)"
    if m != 0 and p == 0:
        return None, None, None, "Camber position cannot be 0 for cambered airfoils"
    return m, p, t, None

# ── Zero-lift AoA estimate ────────────────────────────────────────────────────
def zero_lift_aoa(m, p):
    if m == 0:
        return 0.0
    return -m / (10 * (1 - p/10)**2) * (1 - 2*p/10 + (p/10)**2) * (180/np.pi) * 0.5

# ── Header ────────────────────────────────────────────────────────────────────
st.title("NACA Airfoil Surrogate Model")
st.markdown("""
<div style='background-color:#1e3a5f; padding:14px 16px; border-radius:8px; margin-bottom:1rem; font-size:14px'>
<b>What is this?</b> A Random Forest trained on 20,000+ XFoil simulations across 163 NACA 
4-digit airfoils. Enter any NACA 4-digit code and get instant Cl/Cd predictions — 
compared against the actual XFoil simulation data used for training.
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Airfoil Parameters")


# Sliders
m = st.sidebar.slider("Max Camber (m) — % of chord", 0, 4, 2)
if m == 0:
    p = 0
    st.sidebar.slider("Camber Position (p)", 0, 5, 0, disabled=True)
else:
    p = st.sidebar.slider("Camber Position (p) — tenths of chord", 1, 5, 4)
t = st.sidebar.slider("Thickness (t) — % of chord", 6, 20, 12, step=2)
re = st.sidebar.selectbox("Reynolds Number", [500000, 1000000, 2000000],
                           index=1, format_func=lambda x: f"{x/1e6:.1f}M")

naca_code = f"{m}{p}{t:02d}"

# Airfoil shape in sidebar
st.sidebar.markdown(f"### NACA {naca_code}")
st.sidebar.markdown(make_airfoil_svg(m, p, t), unsafe_allow_html=True)
st.sidebar.markdown(f"""
<small style='color:gray'>
Max camber: <b>{m}%</b> · Position: <b>{p*10}%</b> · Thickness: <b>{t}%</b>
</small>
""", unsafe_allow_html=True)

# ── Predictions ───────────────────────────────────────────────────────────────
aoa_range = np.arange(-10, 15.5, 0.5)
input_df  = pd.DataFrame({'m': m, 'p': p, 't': t, 're': re, 'aoa': aoa_range})
predictions = model.predict(input_df)
cl_pred = predictions[:, 0]
cd_pred = predictions[:, 1]
ld_pred = cl_pred / np.where(cd_pred > 0, cd_pred, np.nan)

ref      = df_ref[(df_ref['naca'] == naca_code) & (df_ref['re'] == re)]
has_ref  = len(ref) > 0

best_ld_idx  = np.nanargmax(ld_pred)
best_ld_aoa  = aoa_range[best_ld_idx]
best_ld_val  = ld_pred[best_ld_idx]
best_ld_cl   = cl_pred[best_ld_idx]
best_ld_cd   = cd_pred[best_ld_idx]
zl_aoa       = zero_lift_aoa(m, p)

# ── Key metrics ───────────────────────────────────────────────────────────────
st.markdown("### Key Performance Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Best L/D",          f"{best_ld_val:.1f}")
c2.metric("AoA at Best L/D",   f"{best_ld_aoa:.1f}°")
c3.metric("Max Cl",            f"{cl_pred.max():.3f}")
c4.metric("Min Cd",            f"{cd_pred.min():.4f}")
c5.metric("Zero-lift AoA",     f"{zl_aoa:.1f}°")

# ── Physics annotations ───────────────────────────────────────────────────────
if m > 0:
    sym_zl = 0.0
    delta_zl = zl_aoa - sym_zl
    st.info(
        f"**NACA {naca_code}** — {m}% camber shifts the zero-lift angle to "
        f"**{zl_aoa:.1f}°** ({abs(delta_zl):.1f}° more negative than a symmetric airfoil). "
        f"The airfoil generates lift at 0° AoA. Optimal efficiency (L/D = {best_ld_val:.0f}) "
        f"is reached at **{best_ld_aoa:.1f}° AoA**, where Cl = {best_ld_cl:.3f} "
        f"and Cd = {best_ld_cd:.4f}."
    )
else:
    st.info(
        f"**NACA {naca_code}** — symmetric airfoil. Zero lift at 0° AoA by definition. "
        f"Optimal efficiency (L/D = {best_ld_val:.0f}) is reached at "
        f"**{best_ld_aoa:.1f}° AoA**, where Cl = {best_ld_cl:.3f} and Cd = {best_ld_cd:.4f}."
    )

if not has_ref:
    st.warning(f"NACA {naca_code} at Re={re/1e6:.1f}M is outside the training dataset — "
               f"showing surrogate prediction only. Accuracy may be lower.")

# ── Plots ─────────────────────────────────────────────────────────────────────
st.markdown("### Surrogate Prediction vs XFoil Data")
col1, col2, col3, col4 = st.columns(4)

def mark_optimum(ax, aoa, val, color):
    ax.axvline(x=aoa, color=color, linewidth=0.8, linestyle='--', alpha=0.5)
    ax.plot(aoa, val, 'o', color=color, markersize=8, zorder=6)
    ax.annotate(
        f" L/D opt\n AoA={aoa:.1f}°",
        xy=(aoa, val),
        fontsize=8,
        color=color,
        va='bottom',
        zorder=7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                  edgecolor=color, alpha=0.85)
    )

# Cl vs AoA
with col1:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(aoa_range, cl_pred, 'b-', linewidth=2, label='Surrogate')
    if has_ref:
        ax.scatter(ref['aoa'], ref['cl'], color='orange', s=15, zorder=5, label='XFoil')
    mark_optimum(ax, best_ld_aoa, best_ld_cl, 'blue')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('AoA (°)'); ax.set_ylabel('Cl')
    ax.set_title(f'Lift curve — NACA {naca_code}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Cd vs AoA
with col2:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(aoa_range, cd_pred, 'r-', linewidth=2, label='Surrogate')
    if has_ref:
        ax.scatter(ref['aoa'], ref['cd'], color='orange', s=15, zorder=5, label='XFoil')
    ax.axvline(x=best_ld_aoa, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.plot(best_ld_aoa, best_ld_cd, 'o', color='red', markersize=8, zorder=6)
    ax.annotate(f" L/D opt\n AoA={best_ld_aoa:.1f}°", xy=(best_ld_aoa, best_ld_cd),
                fontsize=8, color='red', va='bottom', zorder=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='red', alpha=0.85))
    ax.set_xlabel('AoA (°)'); ax.set_ylabel('Cd')
    ax.set_title(f'Drag polar — NACA {naca_code}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# L/D vs AoA
with col3:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(aoa_range, ld_pred, 'g-', linewidth=2, label='Surrogate')
    if has_ref:
        ld_ref = ref['cl'].values / ref['cd'].values
        ax.scatter(ref['aoa'], ld_ref, color='orange', s=15, zorder=5, label='XFoil')
    ax.plot(best_ld_aoa, best_ld_val, 'o', color='green', markersize=8, zorder=6)
    ax.annotate(f" L/D={best_ld_val:.0f}\n AoA={best_ld_aoa:.1f}°",
                xy=(best_ld_aoa, best_ld_val),
                xytext=(best_ld_aoa + 0.5, best_ld_val - 15),
                fontsize=8, color='green', va='top', zorder=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='green', alpha=0.85),
                arrowprops=dict(arrowstyle='->', color='green', lw=0.8))
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('AoA (°)'); ax.set_ylabel('L/D')
    ax.set_title(f'L/D ratio — NACA {naca_code}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Cl/Cd polar (aerodynamicist's view)
with col4:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(cd_pred, cl_pred, 'purple', linewidth=2, label='Surrogate')
    if has_ref:
        ax.scatter(ref['cd'], ref['cl'], color='orange', s=15, zorder=5, label='XFoil')
    ax.plot(best_ld_cd, best_ld_cl, 'o', color='purple', markersize=8, zorder=6)
    ax.annotate(f" L/D opt\n ({best_ld_cd:.4f}, {best_ld_cl:.3f})",
                xy=(best_ld_cd, best_ld_cl), fontsize=8, color='purple', va='bottom',
                zorder=7, bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='purple', alpha=0.85))
    ax.set_xlabel('Cd'); ax.set_ylabel('Cl')
    ax.set_title(f'Cl/Cd polar — NACA {naca_code}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ── Data table ────────────────────────────────────────────────────────────────
with st.expander("View full prediction table"):
    results_df = pd.DataFrame({
        'AoA (°)':         aoa_range,
        'Cl (predicted)':  cl_pred.round(4),
        'Cd (predicted)':  cd_pred.round(4),
        'L/D (predicted)': np.round(ld_pred, 2)
    })
    st.dataframe(results_df, use_container_width=True)