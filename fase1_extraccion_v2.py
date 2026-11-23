# fase 1 (v2): preprocesamiento y extraccion concurrente con descriptores ampliados - apnea-ecg (physionet)
# avance 3: se amplian los 11 descriptores del avance 2 con hrv clasica, poincare, sampen sobre rr,
# edr (respiracion derivada del ecg) y contexto temporal (minutos vecinos)
# requiere: pip install wfdb numpy scipy

if __name__ == "__main__":
    print("Iniciando...", flush=True)

import time
import numpy as np
import wfdb
from pathlib import Path
from multiprocessing import Pool, cpu_count
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import skew, kurtosis
from numpy.lib.stride_tricks import sliding_window_view

# ---------------- configuracion ----------------
SEMILLA = 2024710081
np.random.seed(SEMILLA)
FS = 100                      # hz
VENT_SEG = 60                 # segundos por ventana
N_VENT = FS * VENT_SEG        # 6000 muestras
M_SAMPEN = 2
R_SAMPEN = 0.2                # r = 0.2 * desviacion estandar
FS_INTERP = 4.0               # hz, remuestreo de la serie rr
BANDAS = {"vlf": (0.01, 0.04), "lf": (0.04, 0.15), "hf": (0.15, 0.40)}
P_MAX = 15                    # tope de procesos para la extraccion completa

# registros de entrenamiento con anotaciones (a=apnea, b=limite, c=control)
REGISTROS = (["a%02d" % i for i in range(1, 21)]
             + ["b%02d" % i for i in range(1, 6)]
             + ["c%02d" % i for i in range(1, 11)])
MAX_REGISTROS = None          # None = todos los registros descargados en datos_apnea

# ---------------- nombres de descriptores (el orden debe coincidir con el return del worker) ----------------
NOMBRES_BASE = ["media_rr", "var_rr", "asim_rr", "curt_rr", "zcr",
                "p_vlf", "p_lf", "p_hf", "lf_hf", "sampen", "n_rr"]          # 11 del avance 2
NOMBRES_HRV = ["sdnn", "rmssd", "pnn50", "fc_media"]
NOMBRES_POINCARE = ["sd1", "sd2", "sd_ratio"]
NOMBRES_NOLIN_RR = ["sampen_rr"]
NOMBRES_EDR = ["amp_media", "amp_std", "p_resp"]
NOMBRES_WORKER = (NOMBRES_BASE + NOMBRES_HRV + NOMBRES_POINCARE
                  + NOMBRES_NOLIN_RR + NOMBRES_EDR)                           # 22 por ventana

# contexto temporal: se calcula en el proceso principal (cada worker ve una sola ventana)
COLS_CTX = ["sdnn", "lf_hf", "sampen_rr", "fc_media"]
NOMBRES_CTX = [n + "_ant" for n in COLS_CTX] + [n + "_sig" for n in COLS_CTX]

NOMBRES_FEAT = NOMBRES_WORKER + NOMBRES_CTX                                   # 30 en total
FAMILIAS = {
    "base": NOMBRES_BASE,
    "hrv": NOMBRES_HRV,
    "poincare": NOMBRES_POINCARE,
    "nolineal_rr": NOMBRES_NOLIN_RR,
    "edr": NOMBRES_EDR,
    "contexto": NOMBRES_CTX,
}
N_WORKER = len(NOMBRES_WORKER)


# ---------------- sampen sobre una serie corta (rr), debe ser de nivel superior para pickle ----------------
def sampen_serie(x, m=2, r_fac=0.2):
    n = len(x)
    r = r_fac * np.std(x)

    def contar(mm):
        pl = np.array([x[i:i + mm] for i in range(n - m)])
        d = np.max(np.abs(pl[:, None, :] - pl[None, :, :]), axis=2)
        return (np.sum(d <= r) - len(pl)) / 2      # sin autocoincidencias

    b = contar(m)
    a = contar(m + 1)
    return -np.log(a / b) if (a > 0 and b > 0) else np.nan


# ---------------- worker (debe ser funcion de nivel superior para pickle) ----------------
def extraer_ventana(ventana):
    nan_vec = np.full(N_WORKER, np.nan)

    # picos r y serie rr
    umbral = 0.5 * np.percentile(ventana, 99.5)
    picos, _ = find_peaks(ventana, height=umbral, distance=int(0.3 * FS))
    rr = np.diff(picos) / FS
    if len(rr) < 10:
        return nan_vec
    t_rr = picos[1:] / FS

    # temporales (sobre rr) + zcr (sobre el ecg de la ventana)
    media_rr = np.mean(rr)
    var_rr = np.var(rr)
    asim_rr = skew(rr)
    curt_rr = kurtosis(rr)
    zcr = np.sum(np.diff(np.sign(ventana - np.mean(ventana))) != 0) / len(ventana)

    # frecuenciales: interpolar rr a 4 hz, quitar media, fft
    t_u = np.arange(t_rr[0], t_rr[-1], 1.0 / FS_INTERP)
    if len(t_u) < 16:
        return nan_vec
    rr_u = np.interp(t_u, t_rr, rr)
    rr_u = rr_u - np.mean(rr_u)
    n_f = len(rr_u)
    espec = np.abs(np.fft.rfft(rr_u)) ** 2 / (FS_INTERP * n_f)
    frec = np.fft.rfftfreq(n_f, d=1.0 / FS_INTERP)
    df = frec[1] - frec[0]
    pot = {}
    for nombre, (f_ini, f_fin) in BANDAS.items():
        mask = (frec >= f_ini) & (frec < f_fin)
        pot[nombre] = np.sum(espec[mask]) * df
    lf_hf = pot["lf"] / pot["hf"] if pot["hf"] > 0 else np.nan

    # entropia muestral sobre el ecg de la ventana: o(n^2), el cuello de botella
    x = ventana
    n = len(x)
    r = R_SAMPEN * np.std(x)
    tm = sliding_window_view(x, M_SAMPEN)[:n - M_SAMPEN]
    cont_b = 0
    cont_a = 0
    for i in range(n - M_SAMPEN - 1):
        d = np.max(np.abs(tm[i + 1:] - tm[i]), axis=1)
        cont_b += np.sum(d <= r)
        d3 = np.maximum(d, np.abs(x[i + M_SAMPEN + 1:n] - x[i + M_SAMPEN]))
        cont_a += np.sum(d3 <= r)
    sampen = -np.log(cont_a / cont_b) if (cont_a > 0 and cont_b > 0) else np.nan

    # ---------------- descriptores nuevos (avance 3) ----------------
    # hrv temporal clasica
    d_rr = np.diff(rr)
    sdnn = np.std(rr, ddof=1)
    rmssd = np.sqrt(np.mean(d_rr ** 2))
    pnn50 = np.mean(np.abs(d_rr) > 0.05) * 100
    fc_media = 60.0 / media_rr

    # poincare
    sd1 = np.std(d_rr, ddof=1) / np.sqrt(2)
    sd2 = np.sqrt(max(2 * np.var(rr, ddof=1) - 0.5 * np.var(d_rr, ddof=1), 0))
    sd_ratio = sd1 / sd2 if sd2 > 0 else np.nan

    # sampen sobre la serie rr (serie corta, costo despreciable)
    sampen_rr = sampen_serie(rr, M_SAMPEN, R_SAMPEN)

    # edr: amplitud de los picos r como proxy de la respiracion
    amp_r = ventana[picos][1:]                    # alineada con t_rr
    amp_media = np.mean(amp_r)
    amp_std = np.std(amp_r)
    amp_u = np.interp(t_u, t_rr, amp_r)
    amp_u = amp_u - np.mean(amp_u)
    espec_a = np.abs(np.fft.rfft(amp_u)) ** 2
    m_resp = (frec >= 0.15) & (frec < 0.40)
    m_tot = (frec >= 0.01) & (frec < 0.40)
    tot_a = np.sum(espec_a[m_tot])
    p_resp = np.sum(espec_a[m_resp]) / tot_a if tot_a > 0 else np.nan

    return np.array([media_rr, var_rr, asim_rr, curt_rr, zcr,
                     pot["vlf"], pot["lf"], pot["hf"], lf_hf, sampen, len(rr),
                     sdnn, rmssd, pnn50, fc_media,
                     sd1, sd2, sd_ratio,
                     sampen_rr,
                     amp_media, amp_std, p_resp])


if __name__ == "__main__":
    print("Bibliotecas cargadas. Iniciando procesamiento...", flush=True)

    # ---------------- ingesta, filtrado y ventaneo (secuencial) ----------------
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    lista_vent = []
    lista_etiq = []
    lista_reg = []
    regs = REGISTROS[:MAX_REGISTROS] if MAX_REGISTROS else REGISTROS

    for reg in regs:
        ruta = str(Path(__file__).resolve().parent / "datos_apnea" / reg)

        print(f"{reg}: leyendo señal local...", flush=True)
        rec = wfdb.rdrecord(ruta)

        print(f"{reg}: leyendo anotaciones locales...", flush=True)
        ann = wfdb.rdann(ruta, "apn")

        print(f"{reg}: filtrando y preparando ventanas...", flush=True)
        ecg = rec.p_signal[:, 0]
        ecg = sosfiltfilt(sos, ecg)
        ecg = (ecg - np.mean(ecg)) / np.std(ecg)   # z-score por registro

        for k in range(len(ann.symbol)):
            ini = ann.sample[k]
            fin = ini + N_VENT
            if fin > len(ecg):
                break
            lista_vent.append(ecg[ini:fin])
            lista_etiq.append(1 if ann.symbol[k] == "A" else 0)
            lista_reg.append(reg)
        print(reg, "->", len(lista_vent), "ventanas acumuladas")

    etiquetas = np.array(lista_etiq)
    registros_arr = np.array(lista_reg)
    print("total ventanas:", len(lista_vent), "| apnea:", etiquetas.sum())

    # ---------------- extraccion completa en paralelo ----------------
    P_FINAL = min(cpu_count(), P_MAX)
    t0 = time.perf_counter()
    with Pool(processes=P_FINAL) as pool:
        feats = pool.map(extraer_ventana, lista_vent, chunksize=8)
    print("\nextraccion completa con", P_FINAL, "procesos:",
          round(time.perf_counter() - t0, 1), "s")

    X_w = np.vstack(feats)                          # (k, 22)

    # ---------------- contexto temporal (minutos vecinos, por paciente) ----------------
    idx_ctx = [NOMBRES_WORKER.index(n) for n in COLS_CTX]
    X_ctx = np.full((len(X_w), 2 * len(idx_ctx)), np.nan)
    for reg in np.unique(registros_arr):
        ii = np.where(registros_arr == reg)[0]      # indices en orden cronologico
        b = X_w[ii][:, idx_ctx]
        ant = np.vstack([b[:1], b[:-1]])            # minuto t-1 (en el borde repite el propio)
        sig = np.vstack([b[1:], b[-1:]])            # minuto t+1
        # si el vecino es nan (ventana invalida), se usa el valor de la propia ventana
        ant = np.where(np.isnan(ant), b, ant)
        sig = np.where(np.isnan(sig), b, sig)
        X_ctx[ii] = np.hstack([ant, sig])
    X = np.hstack([X_w, X_ctx])                     # (k, 30)

    # ---------------- filtrado de ventanas invalidas y guardado ----------------
    validas = ~np.isnan(X).any(axis=1)
    print("ventanas validas:", validas.sum(), "de", len(X))
    print("nan por descriptor (para ver cual descarta ventanas):")
    for j, nombre in enumerate(NOMBRES_FEAT):
        n_nan = int(np.isnan(X[:, j]).sum())
        if n_nan > 0:
            print(f"  {nombre}: {n_nan}")

    np.savez("features_apnea_v2.npz", X=X[validas], y=etiquetas[validas],
             registros=registros_arr[validas], nombres=np.array(NOMBRES_FEAT))
    print("guardado: features_apnea_v2.npz")
