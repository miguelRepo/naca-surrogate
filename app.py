import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="NACA Airfoil Surrogate", layout="wide")

# ── Load model & dataset ─────────────────────────────────────────────────────
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

model = train_model()

@st.cache_data
def load_data():
    df = pd.read_csv('data/xfoil_dataset.csv', dtype={'naca': str})
    df['naca'] = df['naca'].str.zfill(4)
    return df

df_ref = load_data()

# ── NACA 4-digit geometry ─────────────────────────────────────────────────────
def naca4_coords(m, p, t, n=150):
    x = np.linspace(0, 1, n)
    yt = 5*(t/100)*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                    + 0.2843*x**3 - 0.1015*x**4)
    if m == 0 or p == 0:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc = np.where(x < p/10,
                      (m/100)/(p/10)**2*(2*(p/10)*x - x**2),
                      (m/100)/(1-p/10)**2*((1-2*p/10)+2*(p/10)*x - x**2))
        dyc = np.where(x < p/10,
                       2*(m/100)/(p/10)**2*((p/10)-x),
                       2*(m/100)/(1-p/10)**2*((p/10)-x))
    theta = np.arctan(dyc)
    xu = x  - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x  + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)
    return xu, yu, xl, yl, x, yc

def make_airfoil_svg(m, p, t, width=220, height=110):
    """Generate a compact SVG of the airfoil for the sidebar."""
    xu, yu, xl, yl, xc, yc = naca4_coords(m, p, t)
    pad_x, pad_y = 10, 8
    w = width - 2*pad_x
    h = height - 2*pad_y
    y_range = 0.28  # show ±0.14 in y

    def tx(x): return pad_x + x * w
    def ty(y): return height/2 - y/y_range * (h/2)

    # Upper surface path
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

    svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <path d="{fill_path}" fill="rgba(55,138,221,0.15)" stroke="none"/>
  {chord_line}
  {camber_path}
  <path d="{upper_path}" fill="none" stroke="#378ADD" stroke-width="1.5"/>
  <path d="{lower_path}" fill="none" stroke="#378ADD" stroke-width="1.5"/>
</svg>"""
    return svg

# ── Hero section ──────────────────────────────────────────────────────────────
st.title("NACA Airfoil Surrogate Model")
st.markdown("""
<div style='background-color:#1e3a5f; padding:16px; border-radius:8px; margin-bottom:16px'>
<b>What is a surrogate model?</b><br>
Running aerodynamic simulations (like XFoil) for thousands of airfoil designs 
is slow — each run takes seconds, and design exploration may require millions. 
A <b>surrogate model</b> is a machine learning model trained on simulation data 
that can predict the same results <b>instantly</b>.<br><br>
This app uses a <b>Random Forest</b> trained on 20,000+ XFoil simulations across 
163 NACA 4-digit airfoils. Adjust the parameters on the left and see aerodynamic 
predictions update in real time — compared against the actual XFoil data.
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Airfoil Parameters")

m = st.sidebar.slider("Max Camber (m) — % of chord", 0, 4, 2)
if m == 0:
    p = 0
    st.sidebar.slider("Camber Position (p)", 0, 5, 0, disabled=True)
else:
    p = st.sidebar.slider("Camber Position (p) — tenths of chord", 1, 5, 4)
t = st.sidebar.slider("Thickness (t) — % of chord", 6, 20, 12, step=2)
re = st.sidebar.selectbox("Reynolds Number",
                           [500000, 1000000, 2000000],
                           index=1,
                           format_func=lambda x: f"{x/1e6:.1f}M")

naca_code = f"{m}{p}{t:02d}"

# ── Airfoil SVG in sidebar ────────────────────────────────────────────────────
st.sidebar.markdown(f"### NACA {naca_code}")
st.sidebar.markdown(make_airfoil_svg(m, p, t), unsafe_allow_html=True)
st.sidebar.markdown(f"""
<small style='color:gray'>
Max camber: <b>{m}%</b> · Position: <b>{p*10}%</b> · Thickness: <b>{t}%</b>
</small>
""", unsafe_allow_html=True)

# ── Predictions ───────────────────────────────────────────────────────────────
aoa_range = np.arange(-10, 15.5, 0.5)
input_df = pd.DataFrame({'m': m, 'p': p, 't': t, 're': re, 'aoa': aoa_range})
predictions = model.predict(input_df)
cl_pred = predictions[:, 0]
cd_pred = predictions[:, 1]
ld_pred = cl_pred / np.where(cd_pred > 0, cd_pred, np.nan)

ref = df_ref[(df_ref['naca'] == naca_code) & (df_ref['re'] == re)]

# ── Key metrics ───────────────────────────────────────────────────────────────
st.markdown("### Key Performance Metrics")
m1, m2, m3, m4 = st.columns(4)
best_ld_idx = np.nanargmax(ld_pred)
m1.metric("Best L/D",        f"{ld_pred[best_ld_idx]:.1f}")
m2.metric("AoA at Best L/D", f"{aoa_range[best_ld_idx]:.1f}°")
m3.metric("Max Cl",          f"{cl_pred.max():.3f}")
m4.metric("Min Cd",          f"{cd_pred.min():.4f}")

# ── Comparison plots ──────────────────────────────────────────────────────────
st.markdown("### Surrogate Prediction vs XFoil Data")
has_ref = len(ref) > 0
if not has_ref:
    st.info(f"No XFoil reference data for NACA {naca_code} at Re={re/1e6:.1f}M — showing surrogate only.")

col1, col2, col3 = st.columns(3)
plot_configs = [
    ('cl', cl_pred, 'Cl', 'Lift Coefficient (Cl)', 'blue',  col1),
    ('cd', cd_pred, 'Cd', 'Drag Coefficient (Cd)', 'red',   col2),
    (None, ld_pred, 'L/D','L/D Ratio',             'green', col3),
]

for ykey, pred, label, ylabel, color, col in plot_configs:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(aoa_range, pred, color=color, linewidth=2, label='Surrogate model')
    if has_ref:
        if ykey is not None:
            ax.scatter(ref['aoa'], ref[ykey], color='orange', s=20, zorder=5, label='XFoil data')
        else:
            ax.scatter(ref['aoa'], ref['cl']/ref['cd'], color='orange', s=20, zorder=5, label='XFoil data')
    ax.set_xlabel('Angle of Attack (°)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{label} — NACA {naca_code}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    col.pyplot(fig)

# ── Data table ────────────────────────────────────────────────────────────────
with st.expander("📊 View full prediction table"):
    results_df = pd.DataFrame({
        'AoA (°)':        aoa_range,
        'Cl (predicted)': cl_pred.round(4),
        'Cd (predicted)': cd_pred.round(4),
        'L/D (predicted)':np.round(ld_pred, 2)
    })
    st.dataframe(results_df, use_container_width=True)