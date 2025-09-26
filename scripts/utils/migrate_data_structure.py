#!/usr/bin/env python3
"""
Script para migrar la estructura de datos actual a la nueva organización propuesta.
Realiza la migración de manera segura con backups y validación.

Uso:
    python3 scripts/utils/migrate_data_structure.py --dry-run    # Ver qué se haría sin ejecutar
    python3 scripts/utils/migrate_data_structure.py --execute   # Ejecutar la migración
    python3 scripts/utils/migrate_data_structure.py --rollback  # Revertir cambios
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add config module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
import config

class DataMigrator:
    def __init__(self):
        self.data_dir = config.get_data_dir()
        self.backup_dir = self.data_dir / f'backup_migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.migration_log = []

    def log_action(self, action, source=None, dest=None, status="PENDING"):
        """Registra una acción de migración"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'source': str(source) if source else None,
            'dest': str(dest) if dest else None,
            'status': status
        }
        self.migration_log.append(entry)

    def create_new_structure(self, dry_run=False):
        """Crear la nueva estructura de directorios"""
        new_dirs = [
            'raw/mseed/continuos/2024',
            'raw/mseed/continuos/2025',
            'raw/mseed/events/2024',
            'raw/mseed/events/2025',
            'raw/mseed/waveforms',
            'raw/mseed/legacy',
            'raw/datasets',
            'processed/mseed/continuos/2024',
            'processed/mseed/continuos/2025',
            'processed/mseed/events/test_100',
            'processed/mseed/events/test_1000',
            'processed/datasets/test',
            'results/gpd/keras',
            'results/gpd/legacy',
            'results/gpd/tflite',
            'results/stalta',
            'results/comparisons/gpd_vs_analyst',
            'results/comparisons/gpd_vs_stalta',
            'analysis/performance',
            'analysis/statistics',
            'analysis/plots',
            'temp/preprocessing',
            'temp/inference',
            'temp/debug'
        ]

        print("=== CREANDO NUEVA ESTRUCTURA DE DIRECTORIOS ===")
        for dir_path in new_dirs:
            full_path = self.data_dir / dir_path
            self.log_action("CREATE_DIR", dest=full_path)

            if not dry_run:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Creado: {full_path}")
            else:
                print(f"[DRY-RUN] Crearía: {full_path}")

    def migrate_files(self, dry_run=False):
        """Migrar archivos a la nueva estructura"""

        migrations = [
            # Migrar archivos MSEED de dataset/test → raw/mseed/events/2024
            {
                'source': self.data_dir / 'dataset' / 'test',
                'dest': self.data_dir / 'raw' / 'mseed' / 'events' / '2024',
                'pattern': '*.mseed',
                'action': 'MOVE_MSEED_TEST_TO_RAW_EVENTS'
            },

            # Migrar archivos MSEED de dataset/waveforms → raw/mseed/waveforms
            {
                'source': self.data_dir / 'dataset' / 'waveforms',
                'dest': self.data_dir / 'raw' / 'mseed' / 'waveforms',
                'pattern': '*.mseed',
                'action': 'MOVE_MSEED_WAVEFORMS_TO_RAW'
            },

            # Migrar waveforms.csv → raw/datasets/
            {
                'source': self.data_dir / 'dataset',
                'dest': self.data_dir / 'raw' / 'datasets',
                'pattern': 'waveforms.csv',
                'action': 'MOVE_WAVEFORMS_CSV_TO_RAW'
            },

            # Migrar archivos MSEED de dataset/test_1000 → processed/mseed/events/test_1000
            {
                'source': self.data_dir / 'dataset' / 'test_1000',
                'dest': self.data_dir / 'processed' / 'mseed' / 'events' / 'test_1000',
                'pattern': '*.mseed',
                'action': 'MOVE_MSEED_TEST1000_TO_PROCESSED'
            },

            # Migrar todos los CSVs de dataset (excepto waveforms.csv) → processed/datasets/test
            {
                'source': self.data_dir / 'dataset',
                'dest': self.data_dir / 'processed' / 'datasets' / 'test',
                'pattern': 'dataset_estratificado_*.csv',
                'action': 'MOVE_DATASET_CSVS_TO_PROCESSED'
            },

            # Migrar mseed/legacy → raw/mseed/legacy
            {
                'source': self.data_dir / 'mseed' / 'legacy',
                'dest': self.data_dir / 'raw' / 'mseed' / 'legacy',
                'pattern': '*.mseed',
                'action': 'MOVE_LEGACY_MSEED_TO_RAW'
            },

            # Migrar mseed/TMP → temp/preprocessing
            {
                'source': self.data_dir / 'mseed' / 'TMP',
                'dest': self.data_dir / 'temp' / 'preprocessing',
                'pattern': '*.mseed',
                'action': 'MOVE_TMP_MSEED_TO_TEMP'
            },

            # Migrar out/ → results/gpd/keras
            {
                'source': self.data_dir / 'out',
                'dest': self.data_dir / 'results' / 'gpd' / 'keras',
                'pattern': '*',
                'action': 'MOVE_OUT_TO_RESULTS_KERAS'
            },

            # Migrar results/ → results/gpd/legacy
            {
                'source': self.data_dir / 'results',
                'dest': self.data_dir / 'results' / 'gpd' / 'legacy',
                'pattern': '*.out',
                'action': 'MOVE_RESULTS_TO_LEGACY'
            }
        ]

        print(f"\n=== MIGRANDO ARCHIVOS ===")

        for migration in migrations:
            source_dir = migration['source']
            dest_dir = migration['dest']
            pattern = migration['pattern']
            action = migration['action']

            if not source_dir.exists():
                print(f"⚠️  Fuente no existe: {source_dir}")
                continue

            # Buscar archivos que coincidan con el patrón
            if pattern == '*':
                files = [f for f in source_dir.iterdir() if f.is_file()]
            else:
                files = list(source_dir.glob(pattern))

            if not files:
                print(f"⚠️  No hay archivos {pattern} en {source_dir}")
                continue

            print(f"\n--- {action} ---")
            print(f"De: {source_dir}")
            print(f"A:  {dest_dir}")
            print(f"Archivos encontrados: {len(files)}")

            for file_path in files:
                dest_file = dest_dir / file_path.name
                self.log_action(action, file_path, dest_file)

                if not dry_run:
                    try:
                        # Crear directorio destino si no existe
                        dest_dir.mkdir(parents=True, exist_ok=True)

                        # Mover archivo
                        shutil.move(str(file_path), str(dest_file))
                        print(f"✓ Movido: {file_path.name}")
                        self.log_action(action, file_path, dest_file, "SUCCESS")

                    except Exception as e:
                        print(f"✗ Error moviendo {file_path.name}: {e}")
                        self.log_action(action, file_path, dest_file, f"ERROR: {e}")
                else:
                    print(f"[DRY-RUN] Movería: {file_path.name}")

    def cleanup_empty_dirs(self, dry_run=False):
        """Eliminar directorios vacíos de la estructura antigua"""
        old_dirs = [
            self.data_dir / 'dataset' / 'test',
            self.data_dir / 'dataset' / 'test_1000',
            self.data_dir / 'dataset' / 'waveforms',
            self.data_dir / 'dataset',
            self.data_dir / 'mseed' / 'TMP',
            self.data_dir / 'mseed' / 'legacy',
            self.data_dir / 'mseed',
            self.data_dir / 'out'
        ]

        print(f"\n=== LIMPIANDO DIRECTORIOS VACÍOS ===")

        for dir_path in old_dirs:
            if not dir_path.exists():
                continue

            try:
                # Verificar si está vacío
                if not any(dir_path.iterdir()):
                    self.log_action("REMOVE_EMPTY_DIR", source=dir_path)

                    if not dry_run:
                        dir_path.rmdir()
                        print(f"✓ Eliminado directorio vacío: {dir_path}")
                    else:
                        print(f"[DRY-RUN] Eliminaría directorio vacío: {dir_path}")
                else:
                    print(f"⚠️  Directorio no vacío, mantenido: {dir_path}")

            except Exception as e:
                print(f"✗ Error eliminando {dir_path}: {e}")

    def create_backup(self, dry_run=False):
        """Crear backup de la estructura actual"""
        print(f"=== CREANDO BACKUP ===")

        if not dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

            # Copiar estructura actual
            for item in self.data_dir.iterdir():
                if item.name.startswith('backup_'):
                    continue  # No hacer backup de backups previos

                dest_path = self.backup_dir / item.name

                try:
                    if item.is_dir():
                        shutil.copytree(item, dest_path)
                        print(f"✓ Backup directorio: {item.name}")
                    else:
                        shutil.copy2(item, dest_path)
                        print(f"✓ Backup archivo: {item.name}")
                except Exception as e:
                    print(f"✗ Error en backup de {item.name}: {e}")

            print(f"✓ Backup creado en: {self.backup_dir}")
        else:
            print(f"[DRY-RUN] Crearía backup en: {self.backup_dir}")

    def save_migration_log(self):
        """Guardar log de migración"""
        log_file = self.data_dir / 'migration_log.json'

        with open(log_file, 'w') as f:
            json.dump(self.migration_log, f, indent=2)

        print(f"✓ Log de migración guardado en: {log_file}")

    def run_migration(self, dry_run=False):
        """Ejecutar migración completa"""
        print("=" * 60)
        print("MIGRACIÓN DE ESTRUCTURA DE DATOS GPD")
        print("=" * 60)
        print(f"Directorio de datos: {self.data_dir}")

        if dry_run:
            print("\n🔍 MODO DRY-RUN - No se realizarán cambios reales")
        else:
            print("\n⚠️  MODO EJECUCIÓN - Se realizarán cambios permanentes")

        # 1. Crear backup
        if not dry_run:
            self.create_backup(dry_run)

        # 2. Crear nueva estructura
        self.create_new_structure(dry_run)

        # 3. Migrar archivos
        self.migrate_files(dry_run)

        # 4. Limpiar directorios vacíos
        self.cleanup_empty_dirs(dry_run)

        # 5. Guardar log
        if not dry_run:
            self.save_migration_log()

        print(f"\n{'=' * 60}")
        print("MIGRACIÓN COMPLETADA")
        print(f"{'=' * 60}")

    def rollback_migration(self):
        """Revertir la migración usando el backup más reciente"""
        # Buscar backup más reciente
        backup_dirs = [d for d in self.data_dir.glob('backup_migration_*') if d.is_dir()]

        if not backup_dirs:
            print("❌ No se encontraron backups para rollback")
            return

        latest_backup = max(backup_dirs, key=lambda x: x.stat().st_mtime)

        print(f"🔄 Realizando rollback desde: {latest_backup}")
        print("⚠️  Esto eliminará la estructura actual y restaurará el backup")

        response = input("¿Continuar? (y/N): ")
        if response.lower() != 'y':
            print("❌ Rollback cancelado")
            return

        # Eliminar estructura actual (excepto backups)
        for item in self.data_dir.iterdir():
            if item.name.startswith('backup_') or item.name == 'migration_log.json':
                continue

            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                print(f"✓ Eliminado: {item}")
            except Exception as e:
                print(f"✗ Error eliminando {item}: {e}")

        # Restaurar desde backup
        for item in latest_backup.iterdir():
            dest_path = self.data_dir / item.name

            try:
                if item.is_dir():
                    shutil.copytree(item, dest_path)
                else:
                    shutil.copy2(item, dest_path)
                print(f"✓ Restaurado: {item.name}")
            except Exception as e:
                print(f"✗ Error restaurando {item.name}: {e}")

        print("✅ Rollback completado")

def main():
    parser = argparse.ArgumentParser(
        description='Migrar estructura de datos GPD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Ver qué cambios se harían sin ejecutarlos
  python3 scripts/utils/migrate_data_structure.py --dry-run

  # Ejecutar la migración (crea backup automáticamente)
  python3 scripts/utils/migrate_data_structure.py --execute

  # Revertir cambios usando backup más reciente
  python3 scripts/utils/migrate_data_structure.py --rollback
        """)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run', action='store_true',
                      help='Mostrar qué cambios se harían sin ejecutarlos')
    group.add_argument('--execute', action='store_true',
                      help='Ejecutar la migración (crea backup automático)')
    group.add_argument('--rollback', action='store_true',
                      help='Revertir migración usando backup más reciente')

    args = parser.parse_args()

    try:
        migrator = DataMigrator()

        if args.dry_run:
            migrator.run_migration(dry_run=True)
        elif args.execute:
            migrator.run_migration(dry_run=False)
        elif args.rollback:
            migrator.rollback_migration()

    except KeyboardInterrupt:
        print(f"\n❌ Migración interrumpida por el usuario")
    except Exception as e:
        print(f"❌ Error en migración: {e}")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())