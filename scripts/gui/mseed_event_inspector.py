#!/usr/bin/env python3
"""
GUI simple para visualización e inspección de archivos miniSEED
Versión simplificada enfocada en visualización y metadata
"""
import os
import sys
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import tkinter as tk
from tkinter import filedialog, messagebox
from obspy import read, UTCDateTime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class TimeHandler:
    """Maneja operaciones relacionadas con tiempo y conversiones."""
    
    @staticmethod
    def format_time_with_ms(dt):
        """Formatea un datetime incluyendo milisegundos."""
        return f"{dt.strftime('%H:%M:%S')}.{int(dt.microsecond/1000):03d}"


class DataProcessor:
    """Maneja la carga y procesamiento de datos miniSEED."""
    
    def __init__(self):
        self.stream = None
        self.original_stream = None  # Mantener copia del stream original
        self.file_starttime = None
        self.file_endtime = None
        self.filename = None
    
    def load_file(self, filename):
        """Carga y procesa un archivo miniSEED."""
        try:
            # Cargar stream original
            original_stream = read(filename)
            self.original_stream = original_stream.copy()
            
            # Stream para visualización (remuestreado)
            stream = original_stream.copy()
            stream.resample(100.0)
            
            self.stream = stream
            self.filename = filename
            self.file_starttime = original_stream[0].stats.starttime
            self.file_endtime = original_stream[0].stats.endtime
            
            # Imprimir metadata completa
            self._print_metadata()
            
            return True
        except Exception as e:
            raise Exception(f"Error al cargar archivo: {str(e)}")
    
    def _print_metadata(self):
        """Imprime toda la metadata relevante del archivo miniSEED."""
        print("\n" + "="*80)
        print(f"METADATA DEL ARCHIVO: {os.path.basename(self.filename)}")
        print("="*80)
        
        # Información general del stream
        print(f"Número de trazas: {len(self.original_stream)}")
        print(f"Tiempo de inicio: {self.file_starttime}")
        print(f"Tiempo de fin: {self.file_endtime}")
        print(f"Duración total: {self.file_endtime - self.file_starttime} segundos")
        
        print("\n" + "-"*60)
        print("INFORMACIÓN DETALLADA POR TRAZA:")
        print("-"*60)
        
        for i, trace in enumerate(self.original_stream):
            stats = trace.stats
            print(f"\nTraza {i+1}:")
            print(f"  Red: {stats.network}")
            print(f"  Estación: {stats.station}")
            print(f"  Ubicación: {stats.location if stats.location else 'Sin ubicación'}")
            print(f"  Canal: {stats.channel}")
            print(f"  Tiempo inicio: {stats.starttime}")
            print(f"  Tiempo fin: {stats.endtime}")
            print(f"  Duración: {stats.endtime - stats.starttime} segundos")
            print(f"  Frecuencia de muestreo: {stats.sampling_rate} Hz")
            print(f"  Delta (periodo de muestreo): {stats.delta} s")
            print(f"  Número de muestras: {stats.npts}")
            print(f"  Formato: {stats._format}")
            
            # Información estadística de los datos
            data = trace.data
            print(f"  Valor mínimo: {data.min()}")
            print(f"  Valor máximo: {data.max()}")
            print(f"  Valor medio: {data.mean():.6f}")
            print(f"  Desviación estándar: {data.std():.6f}")
            
            # Información adicional si está disponible
            if hasattr(stats, 'calib'):
                print(f"  Factor de calibración: {stats.calib}")
            if hasattr(stats, 'mseed'):
                print(f"  Información MSEED adicional:")
                for key, value in stats.mseed.items():
                    print(f"    {key}: {value}")
        
        print("\n" + "="*80)
    
    def get_file_info(self):
        """Retorna información básica del archivo cargado."""
        if not self.stream:
            return None
        
        # Formatear tiempos con milisegundos
        start_time = TimeHandler.format_time_with_ms(self.file_starttime.datetime)
        end_time = TimeHandler.format_time_with_ms(self.file_endtime.datetime)
        
        return {
            'date': self.file_starttime.date,
            'start_time': start_time,
            'end_time': end_time,
            'duration': self.file_endtime - self.file_starttime
        }
    
    def get_all_data(self):
        """Retorna todos los datos del archivo."""
        if not self.stream:
            raise Exception("No hay archivo cargado")
        
        return self.stream


class PlotManager:
    """Maneja la visualización y gráficos."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.fig = None
        self.ax = None
        self.canvas = None
        self.mouse_move_callback = None
    
    def clear_plot(self):
        """Limpia el gráfico actual."""
        for widget in self.parent_frame.winfo_children():
            widget.destroy()
    
    def create_plot(self, stream, filename):
        """Crea un gráfico con todos los datos del stream."""
        self.clear_plot()
        
        # Determinar el número de subplots necesarios
        num_traces = len(stream)
        
        if num_traces == 1:
            self.fig, self.ax = plt.subplots(figsize=(10, 4))
            axes = [self.ax]
        else:
            self.fig, axes = plt.subplots(num_traces, 1, figsize=(10, 2*num_traces), sharex=True)
            if num_traces == 1:
                axes = [axes]
        
        # Graficar cada traza
        for i, trace in enumerate(stream):
            ax = axes[i] if num_traces > 1 else axes[0]
            
            # Convertir tiempo a horas desde el inicio
            start_time = trace.stats.starttime
            times_hours = [(start_time + t).datetime.hour + 
                          (start_time + t).datetime.minute/60.0 + 
                          (start_time + t).datetime.second/3600.0 + 
                          (start_time + t).datetime.microsecond/3600000000.0
                          for t in trace.times()]
            
            ax.plot(times_hours, trace.data, 'b-', linewidth=0.5)
            ax.set_ylabel(f'{trace.stats.channel}\nAmplitud')
            ax.grid(True, alpha=0.3)
            
            # Título solo en el primer subplot
            if i == 0:
                ax.set_title(f'{os.path.basename(filename)} - {trace.stats.network}.{trace.stats.station}')
        
        # Etiqueta del eje X solo en el último subplot
        axes[-1].set_xlabel('Tiempo (horas del día)')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Crear canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        if self.mouse_move_callback:
            self.canvas.mpl_connect('motion_notify_event', self.mouse_move_callback)
        
        self.canvas.draw()
    
    def set_mouse_callback(self, mouse_move_callback):
        """Establece callback para movimiento del mouse."""
        self.mouse_move_callback = mouse_move_callback


class SimpleViewerGUI:
    """Clase principal de la interfaz gráfica simplificada."""
    
    def __init__(self):
        self._setup_environment()
        self._init_components()
        self._create_gui()
        self._setup_callbacks()
    
    def _setup_environment(self):
        """Configura variables de entorno."""
        load_dotenv(find_dotenv())
        self.project_root = os.getenv("PROJECT_LOCAL_ROOT")
        if not self.project_root:
            print("AVISO: PROJECT_LOCAL_ROOT no definido en .env")
            self.project_root = os.getcwd()  # Usar directorio actual como fallback
    
    def _init_components(self):
        """Inicializa los componentes principales."""
        self.data_processor = DataProcessor()
        
    def _create_gui(self):
        """Crea la interfaz gráfica."""
        self.window = tk.Tk()
        self.window.title("Visualizador Simple miniSEED - GPD")
        self.window.geometry("1000x700")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self._create_file_frame()
        self._create_info_frame()
        self._create_actions_frame()
        self._create_plot_frame()
    
    def _create_file_frame(self):
        """Crea el frame para selección de archivo."""
        frame = tk.Frame(self.window)
        frame.pack(fill='x', pady=5)
        
        tk.Label(frame, text="Archivo mseed:", font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        self.entry_archivo = tk.Entry(frame, width=60, font=('Arial', 9))
        self.entry_archivo.pack(side='left', padx=5, fill='x', expand=True)
        
        tk.Button(frame, text="Abrir Archivo...", command=self._open_file, 
                 font=('Arial', 9), bg='lightblue').pack(side='right', padx=5)
    
    def _create_info_frame(self):
        """Crea el frame para información del archivo."""
        frame = tk.Frame(self.window, relief='sunken', bd=1)
        frame.pack(fill='x', pady=5, padx=5)
        
        # Primera fila
        info_frame1 = tk.Frame(frame)
        info_frame1.pack(fill='x', pady=2)
        
        self.lbl_fecha = tk.Label(info_frame1, text="Fecha: --", 
                                 font=('Arial', 10), fg='blue')
        self.lbl_fecha.pack(side='left', padx=10)
        
        # Segunda fila
        info_frame2 = tk.Frame(frame)
        info_frame2.pack(fill='x', pady=2)
        
        self.lbl_inicio = tk.Label(info_frame2, text="Inicio: --", 
                                  font=('Arial', 10), fg='green')
        self.lbl_inicio.pack(side='left', padx=10)
        
        self.lbl_fin = tk.Label(info_frame2, text="Fin: --", 
                               font=('Arial', 10), fg='red')
        self.lbl_fin.pack(side='left', padx=30)
        
        self.lbl_duracion = tk.Label(info_frame2, text="Duración: --", 
                                    font=('Arial', 10), fg='purple')
        self.lbl_duracion.pack(side='left', padx=30)
    
    def _create_actions_frame(self):
        """Crea el frame para acciones y controles."""
        frame = tk.Frame(self.window)
        frame.pack(pady=5)
        
        tk.Button(frame, text="Visualizar Completo", command=self._visualize_full,
                 font=('Arial', 10, 'bold'), bg='lightgreen', padx=20).pack(side='left', padx=10)
        
        tk.Button(frame, text="Limpiar", command=self._clear_plot,
                 font=('Arial', 10), bg='lightyellow', padx=20).pack(side='left', padx=10)
        
        tk.Button(frame, text="Salir", command=self._on_close,
                 font=('Arial', 10), fg='white', bg='red', padx=20).pack(side='right', padx=10)
        
        # Label para posición del mouse
        self.lbl_pos = tk.Label(frame, text="Posición: --", 
                               font=('Arial', 9), fg='gray')
        self.lbl_pos.pack(side='right', padx=20)
    
    def _create_plot_frame(self):
        """Crea el frame para el gráfico."""
        # Frame principal con borde
        main_frame = tk.Frame(self.window, relief='sunken', bd=2)
        main_frame.pack(fill='both', expand=True, pady=5, padx=5)
        
        self.frame_plot = tk.Frame(main_frame)
        self.frame_plot.pack(fill='both', expand=True, pady=2, padx=2)
        
        self.plot_manager = PlotManager(self.frame_plot)
    
    def _setup_callbacks(self):
        """Configura los callbacks."""
        self.plot_manager.set_mouse_callback(self._on_mouse_move)
    
    def _open_file(self):
        """Abre y procesa un archivo miniSEED."""
        initial_dir = os.path.join(self.project_root, "resultados", "mseed") if self.project_root else os.getcwd()
        
        filename = filedialog.askopenfilename(
            title="Selecciona un archivo miniSEED",
            initialdir=initial_dir,
            filetypes=[("MiniSEED", "*.mseed"), ("Todos los archivos", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.data_processor.load_file(filename)
            self._update_file_info()
            
            self.entry_archivo.delete(0, tk.END)
            self.entry_archivo.insert(0, filename)
            
            print(f"\nArchivo cargado exitosamente: {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"Error al cargar archivo: {str(e)}")
    
    def _update_file_info(self):
        """Actualiza la información del archivo en la interfaz."""
        info = self.data_processor.get_file_info()
        if info:
            self.lbl_fecha.config(text=f"Fecha: {info['date']}")
            self.lbl_inicio.config(text=f"Inicio: {info['start_time']}")
            self.lbl_fin.config(text=f"Fin: {info['end_time']}")
            self.lbl_duracion.config(text=f"Duración: {info['duration']:.3f} segundos")
    
    def _visualize_full(self):
        """Visualiza el archivo completo."""
        if not self.data_processor.stream:
            messagebox.showwarning("Aviso", "Primero abre un archivo miniSEED.")
            return
        
        try:
            stream = self.data_processor.get_all_data()
            self.plot_manager.create_plot(stream, self.entry_archivo.get())
            print("Visualización completa generada exitosamente.")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"Error en visualización: {str(e)}")
    
    def _clear_plot(self):
        """Limpia el gráfico."""
        self.plot_manager.clear_plot()
        print("Gráfico limpiado.")
    
    def _on_mouse_move(self, event):
        """Maneja el movimiento del mouse sobre el gráfico."""
        if not event.inaxes or event.xdata is None:
            self.lbl_pos.config(text="Posición: --")
            return
        
        # Convertir coordenada X de vuelta a tiempo
        hour = event.xdata
        time_str = f"{int(hour):02d}:{int((hour % 1) * 60):02d}:{int(((hour % 1) * 60 % 1) * 60):02d}"
        self.lbl_pos.config(text=f"Tiempo: {time_str}")
    
    def _on_close(self):
        """Maneja el cierre de la aplicación."""
        print("Cerrando visualizador miniSEED...")
        self.window.quit()
        self.window.destroy()
        sys.exit(0)
    
    def run(self):
        """Ejecuta la aplicación."""
        print("="*60)
        print("VISUALIZADOR SIMPLE miniSEED - GPD")
        print("="*60)
        print("Iniciando aplicación...")
        self.window.mainloop()


def main():
    """Función principal."""
    app = SimpleViewerGUI()
    app.run()


if __name__ == "__main__":
    main()