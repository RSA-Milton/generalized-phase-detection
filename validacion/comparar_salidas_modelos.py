#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compara eventos entre dos archivos sin cabeceras y sin columna Probability.
Cada archivo debe tener exactamente 4 columnas separadas por espacios en blanco:
Network, Station, Phase, Time

Criterio de coincidencia por defecto:
    Phase + Time truncado a segundos (sin parte decimal).

Se preserva el orden original de las filas de "picks" y se genera un archivo
con las filas que NO encuentran coincidencia en el archivo "anza".

Opcionalmente se puede exigir coincidencia tambien por Network y Station.
"""

import argparse
import pandas as pd

def leer_txt_sin_header(path):
    # Lee archivos sin cabeceras con 4 columnas exactas
    # Ignora lineas que empiezan con '#'
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        comment='#',
        header=None,
        names=["Network","Station","Phase","Time"],
        dtype=str
    )
    # Asegurar strings y truncar Time a segundos
    for col in df.columns:
        df[col] = df[col].astype(str)
    df["Time_trunc"] = df["Time"].str.split(".").str[0]
    # Conservar orden original
    df["__seq"] = range(len(df))
    return df

def main():
    ap = argparse.ArgumentParser(
        description="Comparar dos archivos sin cabeceras y sin Probability por Phase + Time_trunc."
    )
    ap.add_argument("--picks", required=True, help="Ruta al archivo de picks (4 columnas: Network Station Phase Time).")
    ap.add_argument("--anza", required=True, help="Ruta al archivo anza (4 columnas: Network Station Phase Time).")
    ap.add_argument("--out-no-match", default="no_coinciden.txt",
                    help="Archivo de salida con filas de picks que NO coinciden (se respeta el orden original).")
    ap.add_argument("--incluir-network-station", action="store_true",
                    help="Requerir tambien Network y Station en la llave de coincidencia.")
    args = ap.parse_args()

    df_picks = leer_txt_sin_header(args.picks)
    df_anza  = leer_txt_sin_header(args.anza)

    # Definir llaves de comparacion
    keys = ["Phase", "Time_trunc"]
    if args.incluir_network_station:
        keys = ["Network", "Station", "Phase", "Time_trunc"]

    # Preparar tabla de referencia (unicos) para acelerar merge
    ref = df_anza[keys].drop_duplicates()

    merged = df_picks.merge(ref, on=keys, how="left", indicator=True)

    total_picks = len(df_picks)
    num_matches = (merged["_merge"] == "both").sum()
    num_no_match = total_picks - num_matches

    # Filtrar no coincidentes preservando orden
    no_match = merged[merged["_merge"] != "both"].copy()
    no_match.sort_values("__seq", inplace=True)

    # Seleccionar columnas de salida (sin columnas internas)
    cols_out = ["Network","Station","Phase","Time","Time_trunc"]
    no_match[cols_out].to_csv(args.out_no_match, index=False, sep="\t")

    print(f"Total picks: {total_picks}")
    print(f"Coincidencias (por {' + '.join(keys)}): {num_matches}")
    print(f"No coinciden: {num_no_match}")
    print(f"Archivo guardado: {args.out_no_match}")

if __name__ == "__main__":
    main()
