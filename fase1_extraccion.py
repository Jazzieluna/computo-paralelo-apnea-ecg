# fase 1: preprocesamiento y extraccion concurrente - apnea-ecg (physionet)
# requiere: pip install wfdb numpy scipy

if __name__ == "__main__":
    print("Iniciando programa y cargando bibliotecas...", flush=True)

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

# registros de entrenamiento con anotaciones (a=apnea, b=limite, c=control)
REGISTROS = (["a%02d" % i for i in range(1, 21)]
             + ["b%02d" % i for i in range(1, 6)]
             + ["c%02d" % i for i in range(1, 11)])
MAX_REGISTROS = None           # solo a01, que es el registro descargado
N_BENCH = 400                 # ventanas para medir speedup
LISTA_P = [1, 2, 4, 8]        # numero de procesos a probar


# ---------------- worker (debe ser funcion de nivel superior para pickle) ----------------
def extraer_ventana(ventana):
    nan_vec = np.full(11, np.nan)

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

    return np.array([media_rr, var_rr, asim_rr, curt_rr, zcr,
                     pot["vlf"], pot["lf"], pot["hf"], lf_hf, sampen, len(rr)])


NOMBRES_FEAT = ["media_rr", "var_rr", "asim_rr", "curt_rr", "zcr",
                "p_vlf", "p_lf", "p_hf", "lf_hf", "sampen", "n_rr"]


if __name__ == "__main__":
    print("Bibliotecas cargadas. Iniciando procesamiento...", flush=True)

    # ---------------- ingesta, filtrado y ventaneo (secuencial) ----------------
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    lista_vent = []
    lista_etiq = []
    lista_reg = []
    regs = REGISTROS[:MAX_REGISTROS] if MAX_REGISTROS else REGISTROS

    for reg in regs:
        ruta = str(
            Path(__file__).resolve().parent / "datos_apnea" / reg
        )

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

    # ---------------- benchmark de speedup ----------------
    sub = lista_vent[:N_BENCH]
    tiempos = {}
    for P in LISTA_P:
        t0 = time.perf_counter()
        if P == 1:
            _ = [extraer_ventana(v) for v in sub]     # baseline secuencial real
        else:
            with Pool(processes=P) as pool:
                _ = pool.map(extraer_ventana, sub, chunksize=max(1, len(sub) // (P * 4)))
        tiempos[P] = time.perf_counter() - t0
    t1 = tiempos[1]
    print("\nP | tiempo(s) | speedup | eficiencia")
    for P in LISTA_P:
        sp = t1 / tiempos[P]
        print(P, "|", round(tiempos[P], 2), "|", round(sp, 2), "|", round(sp / P, 2))

    # ---------------- extraccion completa en paralelo ----------------
    P_FINAL = min(cpu_count(), max(LISTA_P))
    t0 = time.perf_counter()
    with Pool(processes=P_FINAL) as pool:
        feats = pool.map(extraer_ventana, lista_vent, chunksize=8)
    print("\nextraccion completa con", P_FINAL, "procesos:",
          round(time.perf_counter() - t0, 1), "s")

    X = np.vstack(feats)
    validas = ~np.isnan(X).any(axis=1)
    print("ventanas validas:", validas.sum(), "de", len(X))
    np.savez("features_apnea.npz", X=X[validas], y=etiquetas[validas],
             registros=registros_arr[validas], nombres=np.array(NOMBRES_FEAT))
    print("guardado: features_apnea.npz")