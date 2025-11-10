#!/usr/bin/env python3
"""
Script para crear la estructura de directorios del proyecto GPD desde un archivo de configuración.

Este script lee un archivo de texto que define la estructura de directorios y crea
todos los directorios especificados dentro del GPD_DATA_DIR configurado en .env.

El archivo de configuración debe estar ubicado en: GPD_DATA_DIR/dir_structure.txt

Formato del archivo de configuración:
  - Una ruta de directorio por línea (relativa a GPD_DATA_DIR)
  - Líneas que comienzan con # son comentarios (ignoradas)
  - Líneas vacías son ignoradas
  - Ejemplo: raw/mseed/events/2024

Uso:
    python3 scripts/utils/create_directory_structure.py --config dir_structure.txt --dry-run
    python3 scripts/utils/create_directory_structure.py --config dir_structure.txt --execute
    python3 scripts/utils/create_directory_structure.py --list
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add config module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
import config


class DirectoryStructureCreator:
    """Clase para crear estructura de directorios desde archivo de configuración"""

    def __init__(self, config_file=None):
        """
        Inicializa el creador de estructura de directorios.

        Args:
            config_file: Ruta al archivo de configuración. Si es None, busca en GPD_DATA_DIR/dir_structure.txt
        """
        self.data_dir = config.get_data_dir()

        # Si no se especifica archivo de configuración, buscar en GPD_DATA_DIR
        if config_file is None:
            self.config_file = self.data_dir / 'dir_structure.txt'
        else:
            # Convertir a Path y resolver ruta absoluta si es necesario
            config_path = Path(config_file)
            if config_path.is_absolute():
                self.config_file = config_path
            else:
                # Si es relativa, buscar primero en data_dir, luego en directorio actual
                if (self.data_dir / config_path).exists():
                    self.config_file = self.data_dir / config_path
                else:
                    self.config_file = config_path

        self.directories = []
        self.creation_log = []

    def parse_config_file(self):
        """
        Lee y parsea el archivo de configuración de estructura de directorios.

        Returns:
            list: Lista de rutas de directorios a crear

        Raises:
            FileNotFoundError: Si el archivo de configuración no existe
            ValueError: Si el archivo está vacío o no tiene directorios válidos
        """
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"Archivo de configuración no encontrado: {self.config_file}\n"
                f"Asegúrate de que el archivo existe en la ubicación especificada."
            )

        directories = []

        print(f"📖 Leyendo configuración desde: {self.config_file}")

        with open(self.config_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Eliminar espacios en blanco al inicio/final
                line = line.strip()

                # Ignorar líneas vacías y comentarios
                if not line or line.startswith('#'):
                    continue

                # Validar que no contenga caracteres inválidos
                if any(char in line for char in ['\\', '\0', '\n', '\r']):
                    print(f"⚠️  Línea {line_num}: Caracteres inválidos ignorados: {line}")
                    continue

                # Normalizar separadores de ruta (convertir \ a /)
                normalized_path = line.replace('\\', '/')

                directories.append(normalized_path)

        if not directories:
            raise ValueError(
                f"No se encontraron directorios válidos en {self.config_file}\n"
                f"El archivo debe contener al menos una ruta de directorio válida."
            )

        self.directories = directories
        print(f"✓ Se encontraron {len(directories)} directorios en la configuración")

        return directories

    def validate_directories(self):
        """
        Valida que las rutas de directorios sean válidas y seguras.

        Returns:
            tuple: (directorios_válidos, directorios_inválidos)
        """
        valid = []
        invalid = []

        for dir_path in self.directories:
            # Validaciones de seguridad
            if '..' in dir_path:
                invalid.append((dir_path, "Contiene '..' (path traversal no permitido)"))
                continue

            if dir_path.startswith('/'):
                invalid.append((dir_path, "Ruta absoluta no permitida (debe ser relativa)"))
                continue

            # Validar que no sea solo espacios o caracteres especiales
            if not dir_path or dir_path.isspace():
                invalid.append((dir_path, "Ruta vacía o solo espacios"))
                continue

            valid.append(dir_path)

        if invalid:
            print(f"\n⚠️  Se encontraron {len(invalid)} rutas inválidas:")
            for path, reason in invalid:
                print(f"   ✗ {path}: {reason}")

        return valid, invalid

    def create_directories(self, dry_run=False):
        """
        Crea los directorios especificados en la configuración.

        Args:
            dry_run: Si es True, solo muestra qué se crearía sin crear nada

        Returns:
            dict: Estadísticas de creación (creados, existentes, errores)
        """
        stats = {
            'created': 0,
            'already_exist': 0,
            'errors': 0,
            'skipped': 0
        }

        print(f"\n{'=' * 70}")
        print("CREACIÓN DE ESTRUCTURA DE DIRECTORIOS")
        print(f"{'=' * 70}")
        print(f"Directorio base: {self.data_dir}")
        print(f"Total directorios a crear: {len(self.directories)}")

        if dry_run:
            print("\n🔍 MODO DRY-RUN - No se crearán directorios realmente")
        else:
            print("\n⚙️  MODO EJECUCIÓN - Creando directorios...")

        print(f"\n{'-' * 70}")

        for dir_path in self.directories:
            full_path = self.data_dir / dir_path

            # Verificar si ya existe
            if full_path.exists():
                if full_path.is_dir():
                    stats['already_exist'] += 1
                    status = "EXISTE"
                    symbol = "○"
                else:
                    # Existe pero no es un directorio (es un archivo)
                    stats['errors'] += 1
                    status = "ERROR (archivo con mismo nombre)"
                    symbol = "✗"
                    print(f"{symbol} {dir_path:<50} {status}")
                    self.creation_log.append({
                        'path': str(full_path),
                        'status': 'error',
                        'reason': 'Existe un archivo con el mismo nombre',
                        'timestamp': datetime.now().isoformat()
                    })
                    continue

                if dry_run:
                    print(f"{symbol} {dir_path:<50} {status}")
                continue

            # Crear directorio
            if not dry_run:
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    stats['created'] += 1
                    status = "CREADO"
                    symbol = "✓"
                    print(f"{symbol} {dir_path:<50} {status}")

                    self.creation_log.append({
                        'path': str(full_path),
                        'status': 'created',
                        'timestamp': datetime.now().isoformat()
                    })

                except PermissionError:
                    stats['errors'] += 1
                    status = "ERROR (permisos insuficientes)"
                    symbol = "✗"
                    print(f"{symbol} {dir_path:<50} {status}")

                    self.creation_log.append({
                        'path': str(full_path),
                        'status': 'error',
                        'reason': 'Permisos insuficientes',
                        'timestamp': datetime.now().isoformat()
                    })

                except Exception as e:
                    stats['errors'] += 1
                    status = f"ERROR ({e})"
                    symbol = "✗"
                    print(f"{symbol} {dir_path:<50} {status}")

                    self.creation_log.append({
                        'path': str(full_path),
                        'status': 'error',
                        'reason': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

            else:
                # Dry run
                stats['created'] += 1
                status = "SE CREARÍA"
                symbol = "►"
                print(f"{symbol} {dir_path:<50} {status}")

        return stats

    def print_summary(self, stats, dry_run=False):
        """
        Imprime resumen de la operación.

        Args:
            stats: Diccionario con estadísticas de creación
            dry_run: Si fue una ejecución en modo dry-run
        """
        print(f"\n{'-' * 70}")
        print("RESUMEN")
        print(f"{'-' * 70}")

        if dry_run:
            print(f"Directorios que se crearían: {stats['created']}")
        else:
            print(f"Directorios creados:         {stats['created']}")

        print(f"Directorios ya existentes:   {stats['already_exist']}")

        if stats['errors'] > 0:
            print(f"Errores:                     {stats['errors']} ⚠️")
        else:
            print(f"Errores:                     {stats['errors']}")

        total = stats['created'] + stats['already_exist'] + stats['errors']
        print(f"Total procesado:             {total}")

        print(f"{'-' * 70}")

        if not dry_run and stats['errors'] == 0:
            print("✅ Estructura de directorios creada exitosamente")
        elif not dry_run and stats['errors'] > 0:
            print("⚠️  Estructura creada con algunos errores")
        else:
            print("🔍 Simulación completada - usa --execute para crear los directorios")

    def list_structure(self):
        """
        Lista la estructura de directorios que se crearía, organizada jerárquicamente.
        """
        print(f"\n{'=' * 70}")
        print("ESTRUCTURA DE DIRECTORIOS CONFIGURADA")
        print(f"{'=' * 70}")
        print(f"Archivo de configuración: {self.config_file}")
        print(f"Directorio base: {self.data_dir}")
        print(f"{'=' * 70}\n")

        # Organizar directorios en estructura de árbol
        tree = {}

        for dir_path in sorted(self.directories):
            parts = dir_path.split('/')
            current = tree

            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {}
                current = current[part]

        # Imprimir árbol
        self._print_tree(tree, self.data_dir.name, 0)

        print(f"\n{'=' * 70}")
        print(f"Total: {len(self.directories)} directorios")
        print(f"{'=' * 70}")

    def _print_tree(self, tree, name, level):
        """
        Imprime un árbol de directorios recursivamente.

        Args:
            tree: Diccionario con estructura de árbol
            name: Nombre del nodo actual
            level: Nivel de profundidad
        """
        prefix = "  " * level

        if level == 0:
            print(f"📁 {name}/")
        else:
            print(f"{prefix}├── {name}/")

        for key in sorted(tree.keys()):
            self._print_tree(tree[key], key, level + 1)


def main():
    parser = argparse.ArgumentParser(
        description='Crear estructura de directorios del proyecto GPD desde archivo de configuración',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Ver la estructura que se creará sin ejecutar
  python3 scripts/utils/create_directory_structure.py --dry-run

  # Crear la estructura de directorios
  python3 scripts/utils/create_directory_structure.py --execute

  # Usar un archivo de configuración específico
  python3 scripts/utils/create_directory_structure.py --config mi_estructura.txt --execute

  # Listar la estructura configurada en formato árbol
  python3 scripts/utils/create_directory_structure.py --list

  # Ver qué pasaría con un archivo de configuración en el directorio actual
  python3 scripts/utils/create_directory_structure.py --config ./dir_structure.txt --dry-run

Ubicación del archivo de configuración:
  - Por defecto: GPD_DATA_DIR/dir_structure.txt (configurado en .env)
  - Custom: usa --config para especificar otra ubicación

Formato del archivo de configuración:
  - Una ruta por línea (relativa a GPD_DATA_DIR)
  - Líneas con # son comentarios
  - Ejemplo: raw/mseed/events/2024
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Ruta al archivo de configuración (default: GPD_DATA_DIR/dir_structure.txt)'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--dry-run',
        action='store_true',
        help='Mostrar qué directorios se crearían sin crearlos'
    )
    group.add_argument(
        '--execute',
        action='store_true',
        help='Crear la estructura de directorios'
    )
    group.add_argument(
        '--list',
        action='store_true',
        help='Listar estructura configurada en formato árbol'
    )

    args = parser.parse_args()

    try:
        # Crear instancia del creador
        creator = DirectoryStructureCreator(config_file=args.config)

        # Si solo se quiere listar
        if args.list:
            creator.parse_config_file()
            creator.list_structure()
            return 0

        # Parsear archivo de configuración
        creator.parse_config_file()

        # Validar directorios
        valid_dirs, invalid_dirs = creator.validate_directories()

        if invalid_dirs:
            print(f"\n❌ Se encontraron {len(invalid_dirs)} rutas inválidas")
            print("Corrige el archivo de configuración y vuelve a intentar")
            return 1

        # Actualizar lista de directorios a los válidos
        creator.directories = valid_dirs

        # Crear directorios
        stats = creator.create_directories(dry_run=args.dry_run)

        # Mostrar resumen
        creator.print_summary(stats, dry_run=args.dry_run)

        # Retornar código de salida apropiado
        if stats['errors'] > 0:
            return 1

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return 1

    except KeyboardInterrupt:
        print(f"\n\n❌ Operación interrumpida por el usuario")
        return 1

    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
