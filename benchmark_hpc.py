# fase 2 y 3: benchmarking hpc, curvas de aprendizaje y clasificacion ml
# requiere: pip install numpy scipy scikit-learn matplotlib wfdb

import os
# 1. Bloquear multithreading automatico en Linux (OpenBLAS/MKL) antes de importar numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

import matplotlib
matplotlib.use("Agg")  # sin ventana, solo guardar imagenes
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# Importar funcion worker y constantes de tu Programa 1 (fase1_extraccion.py)
from fase1_extraccion import extraer_ventana, FS, N_VENT, SEMILLA

np.random.seed(SEMILLA)

# ---------------- configuracion ----------------
N_SPLITS = 5
FRACCIONES = [0.2, 0.4, 0.6, 0.8, 1.0]
REPETICIONES_BENCH = 3   # se reporta la mediana de las repeticiones
NOMBRES_MODELOS = [
    "SVM (RBF)",
    "SVM (RBF, balanced)",
    "Random Forest",
    "Random Forest (balanced)",
]


def guardar_csv(ruta, encabezados, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(encabezados)
        w.writerows(filas)
    print(f"[+] Tabla guardada en: '{ruta}'")


# ==============================================================================
# BLOQUE 1: BENCHMARKING DE COMPUTO PARALELO (HPC)
# ==============================================================================
def cargar_ventanas_para_benchmark(n_ventanas=400):
    """
    Lee los registros locales (a01, a02, ...) para extraer ventanas reales de ECG
    con el MISMO preprocesamiento de la Fase 1 (pasa-banda 0.5-40 Hz + z-score).
    """
    carpeta_datos = Path(__file__).resolve().parent / "datos_apnea"
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    ventanas = []

    for idx in range(1, 21):
        reg = f"a{idx:02d}"
        ruta_reg = carpeta_datos / reg
        if not ruta_reg.with_suffix(".hea").exists():
            continue

        rec = wfdb.rdrecord(str(ruta_reg))
        ann = wfdb.rdann(str(ruta_reg), "apn")
        ecg = rec.p_signal[:, 0]
        ecg = sosfiltfilt(sos, ecg)
        ecg = (ecg - np.mean(ecg)) / np.std(ecg)

        for k in range(len(ann.symbol)):
            ini = ann.sample[k]
            fin = ini + N_VENT
            if fin > len(ecg):
                break
            ventanas.append(ecg[ini:fin])
            if len(ventanas) >= n_ventanas:
                return ventanas

    if len(ventanas) == 0:
        # Ruido aleatorio produce <10 picos R y extraer_ventana sale de inmediato,
        # lo que falsearia el benchmark; mejor fallar con un mensaje claro.
        raise FileNotFoundError(
            "No se encontraron registros en 'datos_apnea/'. "
            "Descarga al menos a01 (.hea, .dat, .apn)."
        )
    print(f"[!] Aviso: solo se hallaron {len(ventanas)} ventanas (se pedian {n_ventanas}).")
    return ventanas


def ejecutar_benchmark_hpc(n_bench=400, lista_p=(1, 2, 4, 8, 16)):
    print("=" * 65)
    print("FASE 2: BENCHMARKING DE COMPUTO PARALELO (MULTIPROCESSING NATIVO)")
    print("=" * 65)

    ventanas_sub = cargar_ventanas_para_benchmark(n_bench)
    n_cpu = cpu_count()
    lista_p = [p for p in lista_p if p <= n_cpu]
    print(f"[*] Evaluando extraccion concurrente sobre {len(ventanas_sub)} ventanas ({N_VENT} muestras c/u)")
    print(f"[*] Nucleos logicos detectados: {n_cpu} | P a probar: {lista_p}")
    print(f"[*] Repeticiones por P: {REPETICIONES_BENCH} (se reporta la mediana)\n")

    # calentamiento: evita que la primera medicion cargue con efectos de cache/import
    extraer_ventana(ventanas_sub[0])

    tiempos = {}
    for P in lista_p:
        corridas = []
        for _ in range(REPETICIONES_BENCH):
            t0 = time.perf_counter()
            if P == 1:
                # Ejecucion estrictamente secuencial (monohilo real para T_1)
                _ = [extraer_ventana(v) for v in ventanas_sub]
            else:
                chunk = max(1, len(ventanas_sub) // (P * 4))
                with Pool(processes=P) as pool:
                    _ = pool.map(extraer_ventana, ventanas_sub, chunksize=chunk)
            corridas.append(time.perf_counter() - t0)
        tiempos[P] = float(np.median(corridas))

    t1 = tiempos[1]
    speedups, eficiencias, filas = {}, {}, []

    print(f"{'Procesos (P)':^14} | {'Tiempo Tp (s)':^15} | {'Speedup (Sp)':^14} | {'Eficiencia (Ep)':^15}")
    print("-" * 65)
    for P in lista_p:
        sp = t1 / tiempos[P]
        ep = sp / P
        speedups[P] = sp
        eficiencias[P] = ep
        filas.append([P, round(tiempos[P], 3), round(sp, 3), round(ep * 100, 2)])
        print(f"{P:^14d} | {tiempos[P]:^15.2f} | {sp:^14.2f} | {ep * 100:^14.2f}%")

    guardar_csv("resultados_benchmark_hpc.csv",
                ["P", "tiempo_s", "speedup", "eficiencia_pct"], filas)

    # Graficar Speedup y Eficiencia
    p_arr = np.array(lista_p)
    sp_arr = np.array([speedups[p] for p in lista_p])
    ep_arr = np.array([eficiencias[p] * 100 for p in lista_p])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(p_arr, sp_arr, "o-", color="#1f77b4", lw=2.2, label="Speedup Real ($S_P$)")
    ax1.plot(p_arr, p_arr, "--", color="gray", lw=1.5, label="Speedup Ideal ($S_P = P$)")
    ax1.set_title("Escalamiento de Speedup ($S_P = T_1 / T_P$)")
    ax1.set_xlabel("Número de Procesos ($P$)")
    ax1.set_ylabel("Speedup ($S_P$)")
    ax1.set_xticks(p_arr)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend()

    ax2.plot(p_arr, ep_arr, "s-", color="#2ca02c", lw=2.2, label="Eficiencia Real ($E_P$)")
    ax2.axhline(100, linestyle="--", color="gray", lw=1.5, label="Eficiencia Ideal (100%)")
    ax2.set_title("Eficiencia Computacional ($E_P = S_P / P$)")
    ax2.set_xlabel("Número de Procesos ($P$)")
    ax2.set_ylabel("Eficiencia (%)")
    ax2.set_xticks(p_arr)
    ax2.set_ylim(0, 115)
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("graficas_rendimiento_hpc.png", dpi=300)
    plt.close()
    print("[+] Grafica HPC guardada en: 'graficas_rendimiento_hpc.png'")


# ==============================================================================
# BLOQUE 2: CLASIFICACION ML CLASICO, CV PARALELA Y CURVAS DE APRENDIZAJE
# ==============================================================================
def crear_modelo(nombre):
    """Fabrica a nivel de modulo (las lambdas no se pueden enviar a otros procesos)."""
    cw = "balanced" if "balanced" in nombre else None
    if nombre.startswith("SVM"):
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=cw, random_state=SEMILLA)
    # n_jobs=1: prohibido usar paralelismo de sklearn/joblib
    return RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight=cw,
        random_state=SEMILLA, n_jobs=1
    )


def ajustar_fold(args):
    """
    Worker: entrena y evalua un modelo en un fold (nativo con multiprocessing).
    Si frac < 1, submuestrea el set de entrenamiento de forma ESTRATIFICADA y aleatoria.
    """
    nombre, X, y, train_idx, test_idx, frac, calcular_train = args

    if frac < 1.0:
        sub_idx, _ = train_test_split(
            train_idx, train_size=frac, stratify=y[train_idx], random_state=SEMILLA
        )
    else:
        sub_idx = train_idx

    escalador = StandardScaler()
    X_tr = escalador.fit_transform(X[sub_idx])
    y_tr = y[sub_idx]
    X_te = escalador.transform(X[test_idx])
    y_te = y[test_idx]

    clf = crear_modelo(nombre)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    return {
        "y_te": y_te,
        "preds": preds,
        "f1_tr": f1_score(y_tr, clf.predict(X_tr)) if calcular_train else np.nan,
        "f1_val": f1_score(y_te, preds),
        "n": len(sub_idx),
    }


def mapear_folds(tareas, paralelo):
    if paralelo:
        with Pool(processes=min(len(tareas), cpu_count())) as pool:
            return pool.map(ajustar_fold, tareas, chunksize=1)
    return [ajustar_fold(t) for t in tareas]


def generar_curvas_aprendizaje(X, y, splits):
    """
    Curvas de aprendizaje (rubrica del Avance 2): F1-Score vs. numero de muestras de
    entrenamiento, con submuestreo estratificado y validacion GroupKFold.
    """
    print("\n[*] Generando curvas de aprendizaje...")
    filas_csv = []

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, nombre in zip(axes.flat, NOMBRES_MODELOS):
        tam, f1_tr, f1_val = [], [], []
        for frac in FRACCIONES:
            tareas = [(nombre, X, y, tr, te, frac, True) for tr, te in splits]
            res = mapear_folds(tareas, paralelo=True)
            tam.append(float(np.mean([r["n"] for r in res])))
            f1_tr.append(float(np.mean([r["f1_tr"] for r in res])))
            f1_val.append(float(np.mean([r["f1_val"] for r in res])))
            filas_csv.append([nombre, frac, round(tam[-1]), round(f1_tr[-1], 4), round(f1_val[-1], 4)])

        ax.plot(tam, f1_tr, "o--", color="#d62728", label="Entrenamiento ($F_1$)")
        ax.plot(tam, f1_val, "s-", color="#1f77b4", lw=2, label="Validación GroupKFold ($F_1$)")
        ax.set_title(f"Curva de Aprendizaje - {nombre}")
        ax.set_xlabel("Número de muestras de entrenamiento")
        ax.set_ylabel("F1-Score")
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("curvas_aprendizaje_ml.png", dpi=300)
    plt.close()
    print("[+] Curvas de aprendizaje guardadas en: 'curvas_aprendizaje_ml.png'")
    guardar_csv("resultados_curvas_aprendizaje.csv",
                ["modelo", "fraccion", "n_muestras", "f1_train", "f1_val"], filas_csv)


def ejecutar_clasificacion_ml(ruta_npz="features_apnea.npz"):
    print("\n" + "=" * 65)
    print("FASE 3: ENTRENAMIENTO Y VALIDACION DIAGNOSTICA (GROUP K-FOLD)")
    print("=" * 65)

    if not Path(ruta_npz).is_file():
        raise FileNotFoundError(
            f"No se encontro '{ruta_npz}'. Ejecuta primero fase1_extraccion.py"
        )

    datos = np.load(ruta_npz, allow_pickle=True)
    X = datos["X"]
    y = datos["y"]
    grupos = datos["registros"]
    nombres = datos["nombres"]

    n_pacientes = len(np.unique(grupos))
    if n_pacientes < N_SPLITS:
        raise ValueError(
            f"GroupKFold({N_SPLITS}) necesita al menos {N_SPLITS} pacientes y solo hay {n_pacientes}. "
            "Corre la Fase 1 con mas registros."
        )

    print(f"[*] Archivo cargado: {ruta_npz}")
    print(f"    - Epocas validas (X): {X.shape[0]} ventanas x {X.shape[1]} descriptores")
    print(f"    - Distribucion (y):   Normal={np.sum(y == 0)} | Apnea={np.sum(y == 1)}")
    print(f"    - Sujetos unicos:     {n_pacientes} pacientes")
    print(f"    - Descriptores:       {list(nombres)}")

    splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups=grupos))

    fig_cm, axes_cm = plt.subplots(2, 2, figsize=(10, 8.5))
    filas_ml = []

    for ax, nombre in zip(axes_cm.flat, NOMBRES_MODELOS):
        tareas = [(nombre, X, y, tr, te, 1.0, False) for tr, te in splits]

        # CV secuencial (T_1) y CV con los folds repartidos en procesos (T_P)
        t0 = time.perf_counter()
        res = mapear_folds(tareas, paralelo=False)
        t_seq = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = mapear_folds(tareas, paralelo=True)
        t_par = time.perf_counter() - t0

        y_true = np.concatenate([r["y_te"] for r in res])
        y_pred = np.concatenate([r["preds"] for r in res])
        f1_folds = [r["f1_val"] for r in res]

        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n---------------- {nombre} ----------------")
        print(f"Tiempo CV secuencial: {t_seq:.2f} s | paralelo ({min(N_SPLITS, cpu_count())} procesos): "
              f"{t_par:.2f} s | speedup: {t_seq / t_par:.2f}")
        print(f"Exactitud (Accuracy):  {acc * 100:.2f}%")
        print(f"Sensibilidad (Recall): {rec * 100:.2f}%")
        print(f"F1-Score:              {f1:.4f}  (por fold: {np.mean(f1_folds):.4f} ± {np.std(f1_folds):.4f})")
        print("Matriz de Confusion:")
        print(cm)
        print("\nReporte por clase:")
        print(classification_report(y_true, y_pred, target_names=["Normal (0)", "Apnea (1)"]))

        filas_ml.append([nombre, round(acc, 4), round(rec, 4), round(f1, 4),
                         round(np.mean(f1_folds), 4), round(np.std(f1_folds), 4),
                         round(t_seq, 2), round(t_par, 2), round(t_seq / t_par, 2)])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Apnea"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{nombre}\n(Acc: {acc*100:.1f}% | F1: {f1:.3f})")

    plt.tight_layout()
    plt.savefig("matrices_confusion_ml.png", dpi=300)
    plt.close()
    print("[+] Matrices de confusion guardadas en: 'matrices_confusion_ml.png'")

    guardar_csv("resultados_clasificacion_ml.csv",
                ["modelo", "accuracy", "recall", "f1", "f1_fold_media", "f1_fold_std",
                 "t_cv_seq_s", "t_cv_par_s", "speedup_cv"], filas_ml)

    generar_curvas_aprendizaje(X, y, splits)


if __name__ == "__main__":
    # 1. Benchmark HPC (extraccion de caracteristicas)
    ejecutar_benchmark_hpc(n_bench=400, lista_p=(1, 2, 4, 8, 15))

    # 2. Clasificacion supervisada, CV paralela y curvas de aprendizaje
    ejecutar_clasificacion_ml("features_apnea.npz")