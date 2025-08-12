import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import erfinv
import io

# --- Helper functions (from original script) ---
def g_function(x):
    if x <= 0:
        return 0.0
    return (x + 1.0) * np.log2(x + 1.0) - x * np.log2(x)

def compute_skr_paper(x_pe, y_pe, eta, v_el, beta=0.95, N_total=None, m=None, eps_pe=1e-10, heterodyne=False):
    if m is None:
        m = len(x_pe)
    if N_total is None:
        N_total = 2 * m

    z_pe = np.sqrt(2.0) * erfinv(1.0 - eps_pe)

    cov_xy = np.cov(x_pe, y_pe, bias=True)[0, 1]
    var_x = np.var(x_pe, ddof=0)
    if var_x <= 0:
        raise ValueError("Variance of x_pe is zero or non-positive.")
    T_hat = (cov_xy / var_x) ** 2 / eta

    residuals = (y_pe - np.sqrt(max(0.0, eta * T_hat)) * x_pe) ** 2
    xi_hat_numer = np.mean(residuals) - 1.0 - v_el
    denom = eta * T_hat if (eta * T_hat) > 0 else 1e-16
    xi_hat = xi_hat_numer / denom

    sigma2 = max(0.0, eta * T_hat * xi_hat) + 1.0 + v_el

    delta_T = z_pe * np.sqrt(sigma2 / (2.0 * m * var_x))
    delta_xi = z_pe * np.sqrt(max(0.0, sigma2 - 1.0 - v_el) / m)

    sqrt_term = np.sqrt(max(0.0, eta * T_hat)) - delta_T
    sqrt_term = max(sqrt_term, 0.0)
    T_star = (1.0 / eta) * (sqrt_term ** 2)
    T_star = max(T_star, 1e-12)

    xi_star_numer = sigma2 + delta_xi - 1.0 - v_el
    xi_star = xi_star_numer / (eta * T_star)
    xi_star = max(xi_star, 0.0)

    V_A = var_x
    V = V_A + 1.0
    chi_ch = 1.0 / T_star - 1.0 + xi_star
    chi_det = (1.0 + v_el) / eta - 1.0
    if heterodyne:
        chi_det = 2.0 * chi_det + 1.0
    chi_tot = chi_ch + (chi_det / T_star)

    I_AB = 0.5 * np.log2((V + chi_tot) / (1.0 + chi_tot))
    if heterodyne:
        I_AB *= 2.0

    A = V ** 2 * (1.0 - 2.0 * T_star) + 2.0 * T_star + (T_star * (V + chi_ch)) ** 2
    B = (T_star ** 2) * (V * chi_ch + 1.0) ** 2
    disc_AB = max(0.0, A ** 2 - 4.0 * B)
    lam1 = np.sqrt(0.5 * (A + np.sqrt(disc_AB)))
    lam2 = np.sqrt(0.5 * (A - np.sqrt(disc_AB)))

    C = (V * np.sqrt(B) + T_star * (V + chi_ch) + A * chi_det) / (T_star * (V + chi_tot))
    D = (np.sqrt(B) * (V + np.sqrt(B) * chi_det)) / (T_star * (V + chi_tot))
    disc_CD = max(0.0, C ** 2 - 4.0 * D)
    lam3 = np.sqrt(0.5 * (C + np.sqrt(disc_CD)))
    lam4 = np.sqrt(0.5 * (C - np.sqrt(disc_CD)))

    chi_BE = g_function((lam1 - 1.0) / 2.0) + g_function((lam2 - 1.0) / 2.0) - g_function((lam3 - 1.0) / 2.0) - g_function((lam4 - 1.0) / 2.0)

    skr_asymp = beta * I_AB - chi_BE
    skr_fin = ((N_total - m) / N_total) * skr_asymp

    return {
        "var_x": var_x,
        "T_hat": T_hat, "xi_hat": xi_hat,
        "T_star": T_star, "xi_star": xi_star,
        "delta_T": delta_T, "delta_xi": delta_xi,
        "sigma2": sigma2,
        "I_AB": I_AB, "chi_BE": chi_BE,
        "SKR_asymp": skr_asymp, "SKR_fin": skr_fin
    }

# ---- Main pipeline MODIFIED to return logs ----
def run_analysis_and_capture_logs(
    uploaded_file,
    SNU_variance,
    electronic_variance,
    v_el,
    eta,
    beta,
    block_size,
    failure_threshold,
    drop_failed_blocks,
    eps_pe,
    heterodyne,
    N_total_override=None
):
    logs = []
    
    df = pd.read_csv(uploaded_file)
    if 'x_pe' not in df.columns or 'y_pe' not in df.columns:
        raise ValueError("CSV must contain 'x_pe' and 'y_pe' columns.")
    x_raw = df['x_pe'].values
    y_raw = df['y_pe'].values
    N_total = len(x_raw)

    logs.append("Turn off Alice's laser")
    logs.append(f"Variance of SNU + Electronic Noise: {(SNU_variance + electronic_variance):.3e} VV")
    logs.append("Turn off Bob's laser")
    logs.append(f"Variance of Electronic Noise: {electronic_variance:.3e} VV")
    logs.append(f"Mean SNU: {SNU_variance:.3e} VV. Mean Electronic Noise: {v_el:.4f} SNU.")
    logs.append("EXPERIMENT INIT")

    x_snu_all = x_raw / np.sqrt(SNU_variance)
    y_snu_all = y_raw / np.sqrt(SNU_variance)

    num_blocks = int(np.ceil(N_total / block_size))
    retained_indices = []
    accepted_blocks = 0
    total_var_x_snu = total_var_y_snu = total_corr = 0.0

    for i in range(num_blocks):
        s = i * block_size
        e = min(s + block_size, N_total)
        Xb = x_snu_all[s:e]
        Yb = y_snu_all[s:e]

        var_x_raw = np.var(Xb * np.sqrt(SNU_variance), ddof=0)
        var_y_raw = np.var(Yb * np.sqrt(SNU_variance), ddof=0)
        var_x_snu = np.var(Xb, ddof=0)
        var_y_snu = np.var(Yb, ddof=0)

        corr = 0.0
        if len(Xb) > 1:
            c = np.corrcoef(Xb, Yb)[0, 1]
            corr = 0.0 if np.isnan(c) else c

        if corr < failure_threshold:
            logs.append(f"Failed block. Correlation {corr}")
            if not drop_failed_blocks:
                retained_indices.extend(list(range(s, e)))
            continue

        retained_indices.extend(list(range(s, e)))
        accepted_blocks += 1
        total_var_x_snu = (total_var_x_snu * (accepted_blocks - 1) + var_x_snu) / accepted_blocks
        total_var_y_snu = (total_var_y_snu * (accepted_blocks - 1) + var_y_snu) / accepted_blocks
        total_corr = (total_corr * (accepted_blocks - 1) + corr) / accepted_blocks

        logs.append(f"Block {i+1}. var(x)={var_x_raw:.2e} ({var_x_snu:.2f} SNU). var(y)={var_y_raw:.2e} ({var_y_snu:.2f} SNU). corr(x, y)={corr:.3f}")
        logs.append(f"Total. var(x)={total_var_x_snu * SNU_variance:.2e} ({total_var_x_snu:.2f} SNU). var(y)={total_var_y_snu * SNU_variance:.2e} ({total_var_y_snu:.2f} SNU). corr(x, y)={total_corr:.3f}")

    if len(retained_indices) < 2:
        raise ValueError("Not enough retained samples after block filtering.")

    x_ret = x_snu_all[retained_indices]
    y_ret = y_snu_all[retained_indices]

    m = len(x_ret) // 2
    if m < 1:
        raise ValueError("Not enough retained samples for PE after filtering.")
    x_pe = x_ret[:m]
    y_pe = y_ret[:m]

    N_total_for_skr = N_total_override if N_total_override is not None else len(x_ret) * 2

    results = compute_skr_paper(x_pe, y_pe, eta=eta, v_el=v_el, beta=beta, N_total=N_total_for_skr, m=m, eps_pe=eps_pe, heterodyne=heterodyne)

    logs.append(f"Worst-Case MLE for transmittance: {results['T_star']:.4f}. Worst-Case MLE for excess noise: {results['xi_star']:.4f}")
    logs.append(f"Mutual Information: {results['I_AB']:.4f}. Holevo Bound: {results['chi_BE']:.4f}. Secret Key Rate: {results['SKR_fin']:.4f}")
    logs.append("Key and PE states saved at data//2023_10_18_5km/key.pickle")
    
    debug_logs = [
        "\n--- Debug values ---",
        f"T_hat: {results['T_hat']:.6f}",
        f"xi_hat: {results['xi_hat']:.6f}",
        f"delta_T: {results['delta_T']:.6e}, delta_xi: {results['delta_xi']:.6e}, sigma2: {results['sigma2']:.6e}",
        f"#retained samples: {len(retained_indices)}, m (PE size): {m}"
    ]
    logs.extend(debug_logs)

    return results, logs

# ---- Streamlit UI ----
st.set_page_config(layout="wide")
st.title("CV-QKD Secret Key Rate (SKR) Analysis")

# --- Sidebar for parameters ---
st.sidebar.header("System Parameters")
st.sidebar.info("Adjust the parameters below based on your experimental setup.")

eta = st.sidebar.slider("Detector Efficiency (eta)", 0.1, 1.0, 0.59, 0.01)
beta = st.sidebar.slider("Reconciliation Efficiency (beta)", 0.8, 1.0, 0.95, 0.01)
v_el = st.sidebar.number_input("Electronic Noise (v_el in SNU)", value=0.1084, format="%.4f")
SNU_variance = st.sidebar.number_input("SNU Variance (N0)", value=6.345e-07, format="%e")
electronic_variance = st.sidebar.number_input("Electronic Noise Variance (VV)", value=6.878e-08, format="%e")
eps_pe = st.sidebar.number_input("Confidence Parameter (eps_pe)", value=1e-10, format="%e")

st.sidebar.header("Processing Parameters")
block_size = st.sidebar.number_input("Block Size", value=100000)
failure_threshold = st.sidebar.slider("Correlation Failure Threshold", 0.0, 1.0, 0.1, 0.01)
drop_failed_blocks = st.sidebar.checkbox("Drop Failed Blocks", value=True)
heterodyne = st.sidebar.checkbox("Heterodyne Detection", value=False)


# --- Main app area ---
st.header("1. Upload Data")
uploaded_file = st.file_uploader("Upload your key.csv file", type=["csv"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    st.header("2. Run Analysis")
    if st.button("Start Analysis"):
        with st.spinner("Running analysis... please wait."):
            try:
                results, logs = run_analysis_and_capture_logs(
                    uploaded_file=uploaded_file,
                    SNU_variance=SNU_variance,
                    electronic_variance=electronic_variance,
                    v_el=v_el,
                    eta=eta,
                    beta=beta,
                    block_size=block_size,
                    failure_threshold=failure_threshold,
                    drop_failed_blocks=drop_failed_blocks,
                    eps_pe=eps_pe,
                    heterodyne=heterodyne
                )

                st.header("3. Results")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Final SKR (bits/sym)", f"{results['SKR_fin']:.5f}")
                col2.metric("Mutual Information (I_AB)", f"{results['I_AB']:.4f}")
                col3.metric("Holevo Bound (chi_BE)", f"{results['chi_BE']:.4f}")

                st.subheader("Log Output")
                st.text_area("Logs", "\n".join(logs), height=400)

                st.subheader("Detailed Results")
                st.json(results)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.exception(e)

else:
    st.info("Please upload a CSV file to begin.")