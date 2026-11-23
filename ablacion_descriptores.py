# ablacion de descriptores por familia - apnea-ecg (avance 3)
# compara el conjunto base (11 del avance 2) contra base + cada familia y contra el conjunto completo,
# con el mismo groupkfold por paciente y la validacion cruzada repartida en procesos (multiprocessing nativo)
# requiere: features_apnea_v2.npz (generado por fase1_extraccion_v2.py)

import os
# bloquear multithreading automatico de las bibliotecas numericas antes de importar numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

from fase1_extraccion_v2 import FAMILIAS, SEMILLA

# ---------------- configuracion ----------------
RUTA_NPZ = "features_apnea_v2.npz"
N_SPLITS = 5
MODELOS = ["RF (balanced)", "SVM (balanced)"]
CONJUNTOS = {                                   # nombre -> familias que se incluyen
    "base": ["base"],
    "base+hrv": ["base", "hrv"],
    "base+poincare": ["base", "poincare"],
    "base+sampen_rr": ["base", "nolineal_rr"],
    "base+edr": ["base", "edr"],
    "base+contexto": ["base", "contexto"],
    "completo": ["base", "hrv", "poincare", "nolineal_rr", "edr", "contexto"],
}


# ---------------- worker: entrena y evalua un modelo en un fold ----------------
def ajustar_fold(args):
    nombre, X, y, tr, te = args
    esc = StandardScaler()                      # el escalador se ajusta solo con entrenamiento
    X_tr = esc.fit_transform(X[tr])
    X_te = esc.transform(X[te])
    if nombre.startswith("SVM"):
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                  random_state=SEMILLA)
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced",
                                     random_state=SEMILLA, n_jobs=1)
    clf.fit(X_tr, y[tr])
    preds = clf.predict(X_te)
    return y[te], preds


if __name__ == "__main__":
    if not Path(RUTA_NPZ).is_file():
        raise FileNotFoundError(f"no se encontro '{RUTA_NPZ}'. ejecuta primero fase1_extraccion_v2.py")

    datos = np.load(RUTA_NPZ, allow_pickle=True)
    X = datos["X"]
    y = datos["y"]
    grupos = datos["registros"]
    nombres = list(datos["nombres"])
    print(f"ventanas: {X.shape[0]} | descriptores: {X.shape[1]} | pacientes: {len(np.unique(grupos))}")

    # mismos folds para todos los conjuntos y modelos (comparacion justa)
    splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups=grupos))

    filas = []
    for modelo in MODELOS:
        print(f"\n=== {modelo} ===")
        print(f"{'conjunto':<18}{'n_desc':>7}{'acc':>8}{'recall':>8}{'f1':>8}{'f1 fold (media±std)':>26}{'t(s)':>8}")
        for nom_conj, fams in CONJUNTOS.items():
            cols_nombres = []
            for f in fams:
                cols_nombres += FAMILIAS[f]
            cols = [nombres.index(c) for c in cols_nombres]
            X_sub = X[:, cols]

            tareas = [(modelo, X_sub, y, tr, te) for tr, te in splits]
            t0 = time.perf_counter()
            with Pool(processes=min(N_SPLITS, cpu_count())) as pool:
                res = pool.map(ajustar_fold, tareas, chunksize=1)
            t_cv = time.perf_counter() - t0

            y_true = np.concatenate([r[0] for r in res])
            y_pred = np.concatenate([r[1] for r in res])
            f1_folds = [f1_score(r[0], r[1]) for r in res]
            acc = accuracy_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            print(f"{nom_conj:<18}{len(cols):>7}{acc:>8.4f}{rec:>8.4f}{f1:>8.4f}"
                  f"{np.mean(f1_folds):>16.4f} ± {np.std(f1_folds):.4f}{t_cv:>8.1f}")
            filas.append([modelo, nom_conj, len(cols), round(acc, 4), round(rec, 4), round(f1, 4),
                          round(np.mean(f1_folds), 4), round(np.std(f1_folds), 4), round(t_cv, 2)])

    with open("resultados_ablacion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "conjunto", "n_desc", "accuracy", "recall", "f1",
                    "f1_fold_media", "f1_fold_std", "t_cv_s"])
        w.writerows(filas)
    print("\n[+] tabla guardada en: 'resultados_ablacion.csv'")

    # grafica: f1 por conjunto y modelo
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ancho = 0.38
    pos = np.arange(len(CONJUNTOS))
    for k, modelo in enumerate(MODELOS):
        vals = [fila[5] for fila in filas if fila[0] == modelo]
        ax.bar(pos + k * ancho, vals, ancho, label=modelo)
    ax.set_xticks(pos + ancho * (len(MODELOS) - 1) / 2)
    ax.set_xticklabels(list(CONJUNTOS.keys()), rotation=25, ha="right")
    ax.set_ylabel("F1-Score (apnea)")
    ax.set_title("Ablación de descriptores por familia (GroupKFold por paciente)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig("ablacion_descriptores.png", dpi=300)
    plt.close()
    print("[+] grafica guardada en: 'ablacion_descriptores.png'")
