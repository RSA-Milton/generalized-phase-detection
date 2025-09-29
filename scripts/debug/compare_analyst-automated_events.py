#!/usr/bin/env python
"""
Programa para análisis comparativo entre detecciones del analista humano y métodos automáticos (GPD, STA/LTA, etc.)
Analiza precisión temporal, tasas de detección, errores sistemáticos y métricas P/R/F1

Uso:
python compare_analyst-automated_events.py --analyst dataset_estratificado_1000_with_snr.csv --gpd resultados_evaluacion_1000.csv -V
python compare_analyst-automated_events.py --analyst dataset_estratificado_1000_with_snr.csv --stalta resultados_stalta.csv --tolerance 1.0 2.0 5.0
python compare_analyst-automated_events.py --gpd resultados_gpd.csv --stalta resultados_stalta.csv --output comparacion_metodos.xlsx
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Importar configuración
current_dir = Path(__file__).parent
config_path = current_dir.parent / 'config'
sys.path.insert(0, str(config_path))
import config

class ConsoleLogger:
    """Clase para capturar salida de consola y guardarla en archivo"""
    def __init__(self, filename=None):
        self.terminal = sys.stdout
        self.log_file = None
        if filename:
            self.log_file = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        # Escribir a consola
        self.terminal.write(message)
        # Escribir a archivo si existe
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()

def resolve_file_path(filename, data_type):
    """
    Resuelve la ruta completa del archivo basada en el tipo de datos y la configuración

    Args:
        filename: Nombre del archivo (sin ruta)
        data_type: Tipo de datos ('analyst', 'gpd', 'stalta')

    Returns:
        Path: Ruta completa del archivo
    """
    if data_type == 'analyst':
        # Buscar en subdirectorio test de datasets procesados
        base_dir = config.get_processed_datasets_dir() / 'test'
        file_path = base_dir / filename
    elif data_type == 'gpd':
        # Buscar en directorio de resultados GPD Keras
        base_dir = config.get_results_gpd_keras_dir()
        file_path = base_dir / filename
    elif data_type == 'stalta':
        # Buscar en directorio de resultados STA/LTA
        base_dir = config.get_results_stalta_dir()
        file_path = base_dir / filename
    else:
        raise ValueError(f"Tipo de datos no válido: {data_type}")

    return file_path

def validate_file_inputs(args):
    """
    Valida los archivos de entrada y determina los tipos de comparación

    Returns:
        tuple: (file1_path, file1_type, file2_path, file2_type)
    """
    # Contar cuántos tipos de datos se especificaron
    inputs = []
    if args.analyst:
        inputs.append(('analyst', args.analyst))
    if args.gpd:
        inputs.append(('gpd', args.gpd))
    if args.stalta:
        inputs.append(('stalta', args.stalta))

    if len(inputs) != 2:
        raise ValueError("Debe especificar exactamente dos tipos de datos para comparar")

    # Resolver rutas
    file1_type, file1_name = inputs[0]
    file2_type, file2_name = inputs[1]

    file1_path = resolve_file_path(file1_name, file1_type)
    file2_path = resolve_file_path(file2_name, file2_type)

    # Verificar que los archivos existen
    if not file1_path.exists():
        raise FileNotFoundError(f"Archivo {file1_type} no encontrado: {file1_path}")
    if not file2_path.exists():
        raise FileNotFoundError(f"Archivo {file2_type} no encontrado: {file2_path}")

    return file1_path, file1_type, file2_path, file2_type

def setup_console_logging(output_name):
    """Configura logging de consola a archivo"""
    if output_name:
        # Crear directorio txt/ si no existe
        txt_dir = config.get_results_comparisons_dir() / 'txt'
        txt_dir.mkdir(parents=True, exist_ok=True)

        # Generar ruta completa del archivo de log
        log_filename = txt_dir / f"{output_name}_console.txt"
        logger = ConsoleLogger(str(log_filename))
        sys.stdout = logger
        return logger, str(log_filename)
    return None, None

def parse_timestamp(ts_str):
    """Convierte timestamp a datetime con manejo robusto de formatos"""
    if pd.isna(ts_str) or ts_str == 'NA':
        return None
    
    try:
        # Remover Z y agregar timezone si es necesario
        ts_str = str(ts_str).replace('Z', '+00:00')
        if '+' not in ts_str and 'T' in ts_str:
            ts_str += '+00:00'
        return pd.to_datetime(ts_str)
    except:
        return None

def calculate_ps_difference(t_p, t_s):
    """Calcula diferencia P-S en segundos"""
    if t_p is None or t_s is None:
        return None
    try:
        return (t_s - t_p).total_seconds()
    except:
        return None

def classify_detection_quality(error_p, error_s, threshold_excellent=1.0, threshold_good=5.0):
    """Clasifica calidad de detección basada en errores temporales"""
    if error_p is None and error_s is None:
        return 'Sin detección'
    elif error_p is None or error_s is None:
        return 'Detección parcial'
    
    max_error = max(abs(error_p), abs(error_s))
    if max_error <= threshold_excellent:
        return 'Excelente'
    elif max_error <= threshold_good:
        return 'Buena'
    else:
        return 'Pobre'

def _tp_fp_fn_for_phase(df: pd.DataFrame, phase: str, tau: float) -> Tuple[int,int,int]:
    """Calcula TP, FP, FN para una fase específica con tolerancia tau"""
    if phase == "P":
        ta = df["T-P_analyst_dt"]
        tg = df["T-P_automated_dt"]
    elif phase == "S":
        ta = df["T-S_analyst_dt"]
        tg = df["T-S_automated_dt"]
    else:
        raise ValueError("phase debe ser 'P' o 'S'")

    both = (~ta.isna()) & (~tg.isna())
    only_a = (~ta.isna()) & (tg.isna())
    only_g = (ta.isna()) & (~tg.isna())

    dt = (tg - ta).dt.total_seconds()
    tp = int(((both) & (dt.abs() <= tau)).sum())
    fn = int(only_a.sum() + ((both) & (dt.abs() > tau)).sum())
    fp = int(only_g.sum() + ((both) & (dt.abs() > tau)).sum())

    return tp, fp, fn

def _metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calcula precision, recall y f1 score"""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

def compute_prf1_metrics(df: pd.DataFrame, tau_list: List[float]) -> Dict:
    """Computa métricas P/R/F1 a múltiples niveles"""
    out = {"overall": {}, "per_station": {}}
    
    stations = sorted(df["Estacion"].dropna().unique().tolist())
    
    for tau in tau_list:
        tau_key = f"tau={tau}s"
        
        # Métricas generales
        res_overall = {}
        for phase in ["P", "S"]:
            tp, fp, fn = _tp_fp_fn_for_phase(df, phase, tau)
            res_overall[phase] = _metrics(tp, fp, fn)
        out["overall"][tau_key] = res_overall
        
        # Métricas por estación
        per_st = {}
        macro = {"P": {"precision": [], "recall": [], "f1": []},
                 "S": {"precision": [], "recall": [], "f1": []}}
        
        for st in stations:
            dfs = df[df["Estacion"] == st]
            per_st[st] = {}
            for phase in ["P", "S"]:
                tp, fp, fn = _tp_fp_fn_for_phase(dfs, phase, tau)
                mets = _metrics(tp, fp, fn)
                per_st[st][phase] = mets
                for k in ["precision", "recall", "f1"]:
                    macro[phase][k].append(mets[k])
        
        # Promedio macro
        macro_avg = {ph: {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in macro[ph].items()}
                     for ph in ["P", "S"]}
        
        out["per_station"][tau_key] = {"by_station": per_st, "macro_avg": macro_avg}
    
    return out

def analyze_station_performance(df_comparison, station):
    """Analiza rendimiento específico por estación"""
    station_data = df_comparison[df_comparison['Estacion'] == station].copy()
    
    stats = {
        'total_eventos': len(station_data),
        'detecciones_p': (station_data['T-P_automated'].notna()).sum(),
        'detecciones_s': (station_data['T-S_automated'].notna()).sum(),
        'detecciones_ambas': ((station_data['T-P_automated'].notna()) & (station_data['T-S_automated'].notna())).sum(),
        'tasa_deteccion_p': (station_data['T-P_automated'].notna()).sum() / len(station_data) * 100,
        'tasa_deteccion_s': (station_data['T-S_automated'].notna()).sum() / len(station_data) * 100,
        'inversiones_fisicas': (station_data['inversion_fisica'] == True).sum(),
        'correcciones_aplicadas': (station_data['Corregido'] == True).sum() if 'Corregido' in station_data.columns else 0
    }
    
    # Estadísticas de error temporal para detecciones válidas
    errores_p = station_data['error_temporal_P'].dropna()
    errores_s = station_data['error_temporal_S'].dropna()
    
    if len(errores_p) > 0:
        stats.update({
            'error_p_medio': errores_p.mean(),
            'error_p_std': errores_p.std(),
            'error_p_mediano': errores_p.median(),
            'error_p_rms': np.sqrt((errores_p ** 2).mean())
        })
    
    if len(errores_s) > 0:
        stats.update({
            'error_s_medio': errores_s.mean(),
            'error_s_std': errores_s.std(),
            'error_s_mediano': errores_s.median(),
            'error_s_rms': np.sqrt((errores_s ** 2).mean())
        })
    
    return stats

def generate_comparison_report(df_comparison, tolerance_list, prf1_metrics, output_file=None, verbose=False):
    """Genera reporte detallado de comparación incluyendo métricas P/R/F1"""
    
    print("=" * 80)
    print("REPORTE COMPARATIVO: ANALISTA HUMANO vs MÉTODO AUTOMÁTICO")
    print("=" * 80)
    
    # Estadísticas generales
    total_eventos = len(df_comparison)
    detecciones_p_auto = (df_comparison['T-P_automated'].notna()).sum()
    detecciones_s_auto = (df_comparison['T-S_automated'].notna()).sum()
    detecciones_ambas = ((df_comparison['T-P_automated'].notna()) & (df_comparison['T-S_automated'].notna())).sum()
    
    print(f"\n1. ESTADÍSTICAS GENERALES DE DETECCIÓN")
    print("-" * 50)
    print(f"Total eventos analizados: {total_eventos}")
    print(f"Eventos con P detectada por método automático: {detecciones_p_auto} ({detecciones_p_auto/total_eventos*100:.1f}%)")
    print(f"Eventos con S detectada por método automático: {detecciones_s_auto} ({detecciones_s_auto/total_eventos*100:.1f}%)")
    print(f"Eventos con ambas fases detectadas: {detecciones_ambas} ({detecciones_ambas/total_eventos*100:.1f}%)")
    
    # Analizar inversiones físicas
    if 'inversion_fisica' in df_comparison.columns:
        inversiones = (df_comparison['inversion_fisica'] == True).sum()
        print(f"Inversiones físicas detectadas: {inversiones} ({inversiones/detecciones_ambas*100:.1f}% de eventos con ambas fases)")
    
    # Correcciones aplicadas
    if 'Corregido' in df_comparison.columns:
        correcciones = (df_comparison['Corregido'] == True).sum()
        print(f"Correcciones aplicadas: {correcciones} ({correcciones/total_eventos*100:.1f}%)")
    
    # Análisis de precisión temporal
    print(f"\n2. ANÁLISIS DE PRECISIÓN TEMPORAL")
    print("-" * 50)
    
    errores_p = df_comparison['error_temporal_P'].dropna()
    errores_s = df_comparison['error_temporal_S'].dropna()
    
    if len(errores_p) > 0:
        print(f"ERRORES FASE P ({len(errores_p)} detecciones):")
        print(f"  Media: {errores_p.mean():+.2f} segundos")
        print(f"  Mediana: {errores_p.median():+.2f} segundos")
        print(f"  Desviación estándar: {errores_p.std():.2f} segundos")
        print(f"  RMS: {np.sqrt((errores_p ** 2).mean()):.2f} segundos")
        print(f"  Rango: {errores_p.min():+.2f} a {errores_p.max():+.2f} segundos")
        
        # Distribución de calidad
        errores_p_abs = errores_p.abs()
        excelente_p = (errores_p_abs <= 1.0).sum()
        buena_p = ((errores_p_abs > 1.0) & (errores_p_abs <= 5.0)).sum()
        pobre_p = (errores_p_abs > 5.0).sum()
        
        print(f"  Distribución de calidad:")
        print(f"    Excelente (≤1s): {excelente_p} ({excelente_p/len(errores_p)*100:.1f}%)")
        print(f"    Buena (1-5s): {buena_p} ({buena_p/len(errores_p)*100:.1f}%)")
        print(f"    Pobre (>5s): {pobre_p} ({pobre_p/len(errores_p)*100:.1f}%)")
    
    if len(errores_s) > 0:
        print(f"\nERRORES FASE S ({len(errores_s)} detecciones):")
        print(f"  Media: {errores_s.mean():+.2f} segundos")
        print(f"  Mediana: {errores_s.median():+.2f} segundos")
        print(f"  Desviación estándar: {errores_s.std():.2f} segundos")
        print(f"  RMS: {np.sqrt((errores_s ** 2).mean()):.2f} segundos")
        print(f"  Rango: {errores_s.min():+.2f} a {errores_s.max():+.2f} segundos")
        
        # Distribución de calidad
        errores_s_abs = errores_s.abs()
        excelente_s = (errores_s_abs <= 1.0).sum()
        buena_s = ((errores_s_abs > 1.0) & (errores_s_abs <= 5.0)).sum()
        pobre_s = (errores_s_abs > 5.0).sum()
        
        print(f"  Distribución de calidad:")
        print(f"    Excelente (≤1s): {excelente_s} ({excelente_s/len(errores_s)*100:.1f}%)")
        print(f"    Buena (1-5s): {buena_s} ({buena_s/len(errores_s)*100:.1f}%)")
        print(f"    Pobre (>5s): {pobre_s} ({pobre_s/len(errores_s)*100:.1f}%)")
    
    # Nuevas métricas P/R/F1
    print(f"\n3. MÉTRICAS DE EVALUACIÓN (PRECISION/RECALL/F1)")
    print("-" * 50)
    
    for tau in tolerance_list:
        tau_key = f"tau={tau}s"
        overall_metrics = prf1_metrics["overall"][tau_key]
        
        print(f"\nTolerancia: ±{tau} segundos")
        print("  FASE P:")
        p_metrics = overall_metrics["P"]
        print(f"    Precision: {p_metrics['precision']:.3f}")
        print(f"    Recall:    {p_metrics['recall']:.3f}")
        print(f"    F1-Score:  {p_metrics['f1']:.3f}")
        print(f"    TP/FP/FN:  {p_metrics['tp']}/{p_metrics['fp']}/{p_metrics['fn']}")
        
        print("  FASE S:")
        s_metrics = overall_metrics["S"]
        print(f"    Precision: {s_metrics['precision']:.3f}")
        print(f"    Recall:    {s_metrics['recall']:.3f}")
        print(f"    F1-Score:  {s_metrics['f1']:.3f}")
        print(f"    TP/FP/FN:  {s_metrics['tp']}/{s_metrics['fp']}/{s_metrics['fn']}")
        
        # Promedio macro por estación
        macro_metrics = prf1_metrics["per_station"][tau_key]["macro_avg"]
        print("  PROMEDIO MACRO (por estación):")
        print(f"    Fase P - F1: {macro_metrics['P']['f1']:.3f}, Precision: {macro_metrics['P']['precision']:.3f}, Recall: {macro_metrics['P']['recall']:.3f}")
        print(f"    Fase S - F1: {macro_metrics['S']['f1']:.3f}, Precision: {macro_metrics['S']['precision']:.3f}, Recall: {macro_metrics['S']['recall']:.3f}")
    
    # Análisis por estación
    print(f"\n4. ANÁLISIS POR ESTACIÓN")
    print("-" * 50)
    
    estaciones = sorted(df_comparison['Estacion'].unique())
    station_results = []
    
    for station in estaciones:
        stats = analyze_station_performance(df_comparison, station)
        station_results.append({
            'Estacion': station,
            **stats
        })
        
        print(f"\n{station}:")
        print(f"  Total eventos: {stats['total_eventos']}")
        print(f"  Tasa detección P: {stats['tasa_deteccion_p']:.1f}%")
        print(f"  Tasa detección S: {stats['tasa_deteccion_s']:.1f}%")
        print(f"  Eventos con ambas fases: {stats['detecciones_ambas']} ({stats['detecciones_ambas']/stats['total_eventos']*100:.1f}%)")
        
        if stats.get('error_p_rms'):
            print(f"  Error RMS P: {stats['error_p_rms']:.2f}s")
        if stats.get('error_s_rms'):
            print(f"  Error RMS S: {stats['error_s_rms']:.2f}s")
        
        # Mostrar métricas F1 para tolerancia principal (primera en la lista)
        if tolerance_list and len(tolerance_list) > 0:
            tau_main = tolerance_list[0]
            tau_key = f"tau={tau_main}s"
            station_prf = prf1_metrics["per_station"][tau_key]["by_station"].get(station, {})
            if station_prf:
                p_f1 = station_prf.get("P", {}).get("f1", 0)
                s_f1 = station_prf.get("S", {}).get("f1", 0)
                print(f"  F1-Score (τ={tau_main}s) - P: {p_f1:.3f}, S: {s_f1:.3f}")
        
        if stats['inversiones_fisicas'] > 0:
            print(f"  Inversiones físicas: {stats['inversiones_fisicas']}")
        if stats['correcciones_aplicadas'] > 0:
            print(f"  Correcciones aplicadas: {stats['correcciones_aplicadas']}")
    
    # Casos problemáticos
    print(f"\n5. CASOS PROBLEMÁTICOS")
    print("-" * 50)
    
    # Eventos con errores grandes
    errores_grandes_p = df_comparison[df_comparison['error_temporal_P'].abs() > 10]['mseed'].tolist()
    errores_grandes_s = df_comparison[df_comparison['error_temporal_S'].abs() > 10]['mseed'].tolist()
    
    if errores_grandes_p:
        print(f"Eventos con errores P >10s ({len(errores_grandes_p)}):")
        for evento in errores_grandes_p[:5]:  # Mostrar solo primeros 5
            error = df_comparison[df_comparison['mseed'] == evento]['error_temporal_P'].iloc[0]
            print(f"  {evento}: {error:+.1f}s")
        if len(errores_grandes_p) > 5:
            print(f"  ... y {len(errores_grandes_p)-5} más")
    
    if errores_grandes_s:
        print(f"\nEventos con errores S >10s ({len(errores_grandes_s)}):")
        for evento in errores_grandes_s[:5]:  # Mostrar solo primeros 5
            error = df_comparison[df_comparison['mseed'] == evento]['error_temporal_S'].iloc[0]
            print(f"  {evento}: {error:+.1f}s")
        if len(errores_grandes_s) > 5:
            print(f"  ... y {len(errores_grandes_s)-5} más")
    
    # Guardar reporte detallado si se especifica
    if output_file:
        print(f"\n6. GUARDANDO REPORTE DETALLADO")
        print("-" * 50)

        # Crear directorios necesarios
        comparisons_dir = config.get_results_comparisons_dir()
        xlsx_dir = comparisons_dir / 'xlsx'
        json_dir = comparisons_dir / 'json'
        txt_dir = comparisons_dir / 'txt'

        # Crear directorios si no existen
        xlsx_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        txt_dir.mkdir(parents=True, exist_ok=True)

        # Usar solo el nombre base sin extensiones
        base_name = output_file.replace('.csv', '').replace('.xlsx', '')

        # Crear DataFrame con estadísticas por estación
        df_station_stats = pd.DataFrame(station_results)

        # Crear DataFrames con métricas P/R/F1
        prf1_overall_rows = []
        prf1_macro_rows = []
        prf1_station_rows = []

        for tau in tolerance_list:
            tau_key = f"tau={tau}s"
            # Overall metrics
            for phase in ["P", "S"]:
                metrics = prf1_metrics["overall"][tau_key][phase]
                prf1_overall_rows.append({
                    "tolerancia": tau,
                    "fase": phase,
                    **metrics
                })

            # Macro metrics
            macro_metrics = prf1_metrics["per_station"][tau_key]["macro_avg"]
            for phase in ["P", "S"]:
                prf1_macro_rows.append({
                    "tolerancia": tau,
                    "fase": phase,
                    **macro_metrics[phase]
                })

            # Per station metrics
            by_station = prf1_metrics["per_station"][tau_key]["by_station"]
            for station, phases in by_station.items():
                for phase in ["P", "S"]:
                    prf1_station_rows.append({
                        "tolerancia": tau,
                        "estacion": station,
                        "fase": phase,
                        **phases[phase]
                    })

        df_prf1_overall = pd.DataFrame(prf1_overall_rows)
        df_prf1_macro = pd.DataFrame(prf1_macro_rows)
        df_prf1_station = pd.DataFrame(prf1_station_rows)

        # Guardar múltiples hojas en Excel
        try:
            excel_file = xlsx_dir / f"{base_name}.xlsx"
            with pd.ExcelWriter(excel_file) as writer:
                df_comparison.to_excel(writer, sheet_name='Datos_Completos', index=False)
                df_station_stats.to_excel(writer, sheet_name='Estadisticas_Estacion', index=False)
                df_prf1_overall.to_excel(writer, sheet_name='PRF1_General', index=False)
                df_prf1_macro.to_excel(writer, sheet_name='PRF1_Macro', index=False)
                df_prf1_station.to_excel(writer, sheet_name='PRF1_Por_Estacion', index=False)
            print(f"Reporte Excel guardado en: {excel_file}")

            # Guardar también métricas en JSON
            json_file = json_dir / f"{base_name}_metrics.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(prf1_metrics, f, ensure_ascii=False, indent=2)
            print(f"Métricas JSON guardadas en: {json_file}")

        except Exception as e:
            print(f"Error guardando Excel: {e}")
            # Fallback a CSVs individuales en directorio xlsx
            csv_dir = xlsx_dir  # Usar mismo directorio que Excel
            df_comparison.to_csv(csv_dir / f"{base_name}_datos.csv", index=False)
            df_prf1_overall.to_csv(csv_dir / f"{base_name}_prf1_general.csv", index=False)
            df_prf1_macro.to_csv(csv_dir / f"{base_name}_prf1_macro.csv", index=False)
            df_prf1_station.to_csv(csv_dir / f"{base_name}_prf1_estacion.csv", index=False)
            print(f"Reportes guardados como CSVs en: {csv_dir}")

            # Intentar guardar JSON aunque falle Excel
            try:
                json_file = json_dir / f"{base_name}_metrics.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(prf1_metrics, f, ensure_ascii=False, indent=2)
                print(f"Métricas JSON guardadas en: {json_file}")
            except Exception as json_e:
                print(f"Error guardando JSON: {json_e}")
    
    print(f"\n" + "=" * 80)
    print("FIN DEL REPORTE COMPARATIVO")
    print("=" * 80)
    
    return df_comparison

def main():
    parser = argparse.ArgumentParser(
        description='Análisis comparativo entre diferentes métodos de detección con métricas P/R/F1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tipos de datos soportados:
  --analyst:  Datos del analista humano (busca en GPD_PROCESSED_DATASETS_DIR/test/)
  --gpd:      Resultados de GPD Keras (busca en GPD_RESULTS_GPD_KERAS_DIR)
  --stalta:   Resultados de STA/LTA (busca en GPD_RESULTS_STALTA_DIR)

Ejemplos de uso:
  # Comparar analista vs GPD
  python compare_analyst-automated_events.py --analyst dataset_estratificado_1000_with_snr.csv --gpd resultados_evaluacion_1000.csv -V

  # Comparar analista vs STA/LTA con múltiples tolerancias
  python compare_analyst-automated_events.py --analyst dataset_estratificado_1000_with_snr.csv --stalta resultados_stalta.csv --tolerance 0.5 1.0 2.0 5.0

  # Comparar GPD vs STA/LTA con reporte detallado (solo nombre, sin ruta)
  python compare_analyst-automated_events.py --gpd resultados_gpd.csv --stalta resultados_stalta.csv --output comparacion_metodos

  # Filtrar estaciones específicas con reporte
  python compare_analyst-automated_events.py --analyst dataset.csv --gpd resultados.csv --stations PORT CHAI TENG --tolerance 1.0 2.0 --output analisis_estaciones

Nota: Los reportes se guardan automáticamente en:
  - Excel: GPD_RESULTS_COMPARISONS_DIR/xlsx/nombre.xlsx
  - JSON:  GPD_RESULTS_COMPARISONS_DIR/json/nombre_metrics.json
  - Log:   GPD_RESULTS_COMPARISONS_DIR/txt/nombre_console.txt
        """)

    # Crear grupo mutuamente excluyente para los tipos de datos
    data_group = parser.add_argument_group('Archivos de datos (especificar exactamente 2)')
    data_group.add_argument('--analyst', type=str, default=None,
                           help='Nombre del archivo CSV con datos del analista humano')
    data_group.add_argument('--gpd', type=str, default=None,
                           help='Nombre del archivo CSV con resultados de GPD')
    data_group.add_argument('--stalta', type=str, default=None,
                           help='Nombre del archivo CSV con resultados de STA/LTA')

    parser.add_argument('--output', type=str, default=None,
                       help='Nombre base para reporte detallado (sin ruta, se guarda en GPD_RESULTS_COMPARISONS_DIR)')
    parser.add_argument('--stations', nargs='+', default=None,
                       help='Estaciones específicas a analizar')
    parser.add_argument('--tolerance', nargs='+', type=float, default=[1.0],
                       help='Tolerancias temporales en segundos para métricas P/R/F1 (por defecto: 1.0)')
    parser.add_argument('-V', '--verbose', action='store_true',
                       help='Mostrar información detallada')

    args = parser.parse_args()

    # Configurar logging de consola si se especifica archivo de salida
    console_logger, log_filename = setup_console_logging(args.output)

    try:
        print("=== ANÁLISIS COMPARATIVO ENTRE MÉTODOS DE DETECCIÓN CON MÉTRICAS P/R/F1 ===")

        # Validar y resolver rutas de archivos
        try:
            file1_path, file1_type, file2_path, file2_type = validate_file_inputs(args)
        except (ValueError, FileNotFoundError) as e:
            print(f"ERROR: {e}")
            print("\nDebes especificar exactamente dos tipos de datos:")
            print("  --analyst:  Para datos del analista humano")
            print("  --gpd:      Para resultados de GPD")
            print("  --stalta:   Para resultados de STA/LTA")
            print("\nEjemplo: python compare_analyst-automated_events.py --analyst dataset.csv --gpd resultados.csv")
            return

        print(f"Comparando: {file1_type.upper()} vs {file2_type.upper()}")
        print(f"Archivo {file1_type}: {file1_path}")
        print(f"Archivo {file2_type}: {file2_path}")
        print(f"Tolerancias temporales: {args.tolerance} segundos")
        if log_filename:
            print(f"Log de consola: {log_filename}")

        # Cargar datos
        try:
            df_file1 = pd.read_csv(file1_path)
            df_file2 = pd.read_csv(file2_path)
            print(f"Datos cargados: {len(df_file1)} eventos ({file1_type}), {len(df_file2)} eventos ({file2_type})")
        except Exception as e:
            print(f"ERROR cargando archivos: {e}")
            return

        # Preparar datos para merge basado en los tipos
        # Determinar cuál dataset será el "analyst" y cuál el "automated" para el análisis
        if file1_type == 'analyst':
            df_analyst = df_file1
            df_automated = df_file2
            analyst_type = file1_type
            automated_type = file2_type
        elif file2_type == 'analyst':
            df_analyst = df_file2
            df_automated = df_file1
            analyst_type = file2_type
            automated_type = file1_type
        else:
            # Comparación entre dos métodos automáticos, usar el primero como "reference"
            df_analyst = df_file1
            df_automated = df_file2
            analyst_type = f"{file1_type}_reference"
            automated_type = file2_type
            print(f"Nota: Comparando dos métodos automáticos. {file1_type} será usado como referencia.")

        # Merge de datos por archivo mseed
        # Determinar columnas disponibles para el merge
        analyst_cols = ['Estacion', 'mseed']
        if 'T-P' in df_analyst.columns:
            analyst_cols.extend(['T-P', 'T-S'])
        if 'Pond T-P' in df_analyst.columns:
            analyst_cols.extend(['Pond T-P', 'Pond T-S'])

        df_comparison = pd.merge(
            df_analyst[analyst_cols],
            df_automated,
            on=['Estacion', 'mseed'],
            how='inner',
            suffixes=('_analyst', '_automated')
        )

        print(f"Eventos coincidentes para comparación: {len(df_comparison)}")

        if len(df_comparison) == 0:
            print("ERROR: No hay eventos coincidentes entre los archivos")
            return

        # Filtrar estaciones si se especifica
        if args.stations:
            df_comparison = df_comparison[df_comparison['Estacion'].isin(args.stations)]
            print(f"Eventos después de filtrar estaciones: {len(df_comparison)}")

        # Parsear timestamps
        print("Procesando timestamps...")
        df_comparison['T-P_analyst_dt'] = df_comparison['T-P_analyst'].apply(parse_timestamp)
        df_comparison['T-S_analyst_dt'] = df_comparison['T-S_analyst'].apply(parse_timestamp)
        df_comparison['T-P_automated_dt'] = df_comparison['T-P_automated'].apply(parse_timestamp)
        df_comparison['T-S_automated_dt'] = df_comparison['T-S_automated'].apply(parse_timestamp)

        # Calcular errores temporales
        df_comparison['error_temporal_P'] = df_comparison.apply(
            lambda row: (row['T-P_automated_dt'] - row['T-P_analyst_dt']).total_seconds()
            if row['T-P_automated_dt'] and row['T-P_analyst_dt'] else None, axis=1
        )

        df_comparison['error_temporal_S'] = df_comparison.apply(
            lambda row: (row['T-S_automated_dt'] - row['T-S_analyst_dt']).total_seconds()
            if row['T-S_automated_dt'] and row['T-S_analyst_dt'] else None, axis=1
        )

        # Calcular diferencias P-S usando vectorización
        # Para analista/referencia
        mask_both_analyst = pd.notna(df_comparison['T-P_analyst_dt']) & pd.notna(df_comparison['T-S_analyst_dt'])
        df_comparison['diff_PS_analyst'] = pd.NA
        df_comparison.loc[mask_both_analyst, 'diff_PS_analyst'] = (
            df_comparison.loc[mask_both_analyst, 'T-S_analyst_dt'] -
            df_comparison.loc[mask_both_analyst, 'T-P_analyst_dt']
        ).dt.total_seconds()

        # Para método automático/comparado
        mask_both_auto = pd.notna(df_comparison['T-P_automated_dt']) & pd.notna(df_comparison['T-S_automated_dt'])
        df_comparison['diff_PS_automated'] = pd.NA
        df_comparison.loc[mask_both_auto, 'diff_PS_automated'] = (
            df_comparison.loc[mask_both_auto, 'T-S_automated_dt'] -
            df_comparison.loc[mask_both_auto, 'T-P_automated_dt']
        ).dt.total_seconds()

        # Detectar inversiones físicas
        df_comparison['inversion_fisica'] = df_comparison['diff_PS_automated'] < 0

        # Clasificar calidad de detección
        df_comparison['calidad_deteccion'] = df_comparison.apply(
            lambda row: classify_detection_quality(row['error_temporal_P'], row['error_temporal_S']), axis=1
        )

        # Calcular métricas P/R/F1
        print("Calculando métricas Precision/Recall/F1...")
        prf1_metrics = compute_prf1_metrics(df_comparison, args.tolerance)

        # Generar reporte
        print(f"\nGenerando reporte comparativo: {analyst_type.upper()} vs {automated_type.upper()}")
        generate_comparison_report(df_comparison, args.tolerance, prf1_metrics, args.output, args.verbose)

    finally:
        # Restaurar stdout y cerrar archivo de log
        if console_logger:
            sys.stdout = console_logger.terminal
            console_logger.close()
            if log_filename:
                print(f"\n✓ Log de consola guardado en: {log_filename}")

if __name__ == "__main__":
    main()