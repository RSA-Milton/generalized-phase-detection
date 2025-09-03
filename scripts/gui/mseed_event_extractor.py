#!/usr/bin/env python3
"""
GUI para selección y visualización de eventos de un archivo miniSEED - Versión Refactorizada
"""
import os
import sys
from datetime import datetime, timedelta, time as dt_time
from dotenv import load_dotenv, find_dotenv
import tkinter as tk
from tkinter import filedialog, messagebox
from obspy import read, UTCDateTime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


class TimeHandler:
    """Maneja todas las operaciones relacionadas con tiempo y conversiones."""
    
    @staticmethod
    def parse_time_with_ms(time_str):
        """Parsea una cadena de tiempo que puede incluir milisegundos."""
        try:
            if ',' in time_str:
                main, ms = time_str.split(',')
                t = datetime.strptime(main, '%H:%M:%S').time()
                return dt_time(t.hour, t.minute, t.second, int(ms) * 1000)
            else:
                return datetime.strptime(time_str, '%H:%M:%S').time()
        except Exception as e:
            raise ValueError(f"Formato de hora inválido: {time_str}")
    
    @staticmethod
    def format_time_with_ms(dt):
        """Formatea un datetime incluyendo milisegundos."""
        return f"{dt.strftime('%H:%M:%S')},{int(dt.microsecond/1000):02d}"
    
    @staticmethod
    def create_utc_datetime(base_date, time_obj, shift_seconds=0):
        """Crea un UTCDateTime combinando fecha base, tiempo y desplazamiento."""
        shift_delta = timedelta(seconds=shift_seconds)
        base_dt = datetime.combine(base_date, time_obj) + shift_delta
        return UTCDateTime(base_dt), base_dt


class DataProcessor:
    """Maneja la carga y procesamiento de datos miniSEED."""
    
    def __init__(self):
        self.stream = None
        self.file_starttime = None
        self.file_endtime = None
    
    def load_file(self, filename):
        """Carga y procesa un archivo miniSEED."""
        try:
            stream = read(filename)
            stream.resample(100.0)
            self.stream = stream
            self.file_starttime = stream[0].stats.starttime
            self.file_endtime = stream[0].stats.endtime
            return True
        except Exception as e:
            raise Exception(f"Error al cargar archivo: {str(e)}")
    
    def get_file_info(self):
        """Retorna información básica del archivo cargado."""
        if not self.stream:
            return None
        
        return {
            'date': self.file_starttime.date,
            'start_time': self.file_starttime.datetime.strftime('%H:%M:%S'),
            'end_time': self.file_endtime.datetime.strftime('%H:%M:%S')
        }
    
    def extract_segment(self, start_time, duration, channel):
        """Extrae un segmento de datos específico."""
        if not self.stream:
            raise Exception("No hay archivo cargado")
        
        end_time = start_time + duration
        segment = self.stream.slice(starttime=start_time, endtime=end_time).select(channel=channel)
        
        if not segment:
            raise Exception("No hay datos en el intervalo especificado")
        
        return segment

    def extract_segment_multi(self, start_time, duration):
        """Devuelve un Stream con TODOS los canales en la ventana [start_time, start_time+duration]."""
        end_time = start_time + duration
        # Recorte de todo el stream, sin select(channel=...)
        segment = self.stream.slice(starttime=start_time, endtime=end_time)
        # Asegurar recorte a frontera de muestra
        segment.trim(starttime=start_time, endtime=end_time, nearest_sample=True)
        if not segment:
                raise Exception("No hay datos en el intervalo especificado")
        return segment


class PlotManager:
    """Maneja la visualización y gráficos."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.fig = None
        self.ax = None
        self.canvas = None
        self.click_callback = None
        self.mouse_move_callback = None
    
    def clear_plot(self):
        """Limpia el gráfico actual."""
        for widget in self.parent_frame.winfo_children():
            widget.destroy()
    
    def create_plot(self, segment, duration, filename):
        """Crea un nuevo gráfico con el segmento de datos."""
        self.clear_plot()
        
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        
        # Graficar todos los trazos del segmento
        for trace in segment:
            self.ax.plot(trace.times(), trace.data, label=trace.stats.channel)
        
        # Línea central
        center = duration / 2.0
        self.ax.axvline(center, color='r', linewidth=1, alpha=0.7)
        
        # Configurar ejes y etiquetas
        self.ax.set_xlabel('Tiempo (s)')
        self.ax.set_ylabel('Amplitud')
        self.ax.set_title(os.path.basename(filename))
        self.ax.set_xlim(0, duration)
        self.ax.legend(loc='upper right', fontsize='small')
        
        # Crear canvas y conectar eventos
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        if self.mouse_move_callback:
            self.canvas.mpl_connect('motion_notify_event', self.mouse_move_callback)
        if self.click_callback:
            self.canvas.mpl_connect('button_press_event', self.click_callback)
        
        self.canvas.draw()
    
    def set_callbacks(self, click_callback=None, mouse_move_callback=None):
        """Establece callbacks para eventos del mouse."""
        self.click_callback = click_callback
        self.mouse_move_callback = mouse_move_callback


class CenteringMode:
    """Maneja el modo de centrado de eventos."""
    
    def __init__(self, button, position_label):
        self.is_active = False
        self.button = button
        self.position_label = position_label
    
    def toggle(self):
        """Alterna el modo de centrado."""
        self.is_active = not self.is_active
        self._update_ui()
    
    def deactivate(self):
        """Desactiva el modo de centrado."""
        self.is_active = False
        self._update_ui()
    
    def _update_ui(self):
        """Actualiza la interfaz según el estado del modo."""
        if self.is_active:
            self.button.config(relief=tk.SUNKEN, text='Centrar: ON')
            self.position_label.config(text='Δ: --')
        else:
            self.button.config(relief=tk.RAISED, text='Centrar: OFF')
            self.position_label.config(text='Posición: --')


class KeepCenterMode:
    """Maneja el modo de mantener centro fijo al cambiar duración."""
    
    def __init__(self, button, time_handler):
        self.is_active = False
        self.button = button
        self.time_handler = time_handler
        self.last_center_time = None  # Guarda el tiempo del último centro calculado
        self.last_base_date = None    # Guarda la fecha base del archivo
    
    def toggle(self):
        """Alterna el modo mantener centro."""
        self.is_active = not self.is_active
        self._update_ui()
    
    def deactivate(self):
        """Desactiva el modo mantener centro."""
        self.is_active = False
        self.last_center_time = None
        self.last_base_date = None
        self._update_ui()
    
    def set_center_reference(self, center_utc_time, base_date):
        """Establece el tiempo de referencia del centro actual."""
        self.last_center_time = center_utc_time
        self.last_base_date = base_date
    
    def calculate_new_start_time(self, new_duration):
        """Calcula la nueva hora de inicio para mantener el centro fijo."""
        if not self.is_active or not self.last_center_time or not self.last_base_date:
            return None
        
        # El nuevo inicio debe ser: centro - (nueva_duración / 2)
        half_duration = new_duration / 2.0
        new_start_utc = self.last_center_time - half_duration
        new_start_dt = new_start_utc.datetime
        
        # Extraer solo la parte de tiempo (sin fecha)
        new_start_time = dt_time(
            new_start_dt.hour, 
            new_start_dt.minute, 
            new_start_dt.second, 
            new_start_dt.microsecond
        )
        
        return new_start_time
    
    def _update_ui(self):
        """Actualiza la interfaz según el estado del modo."""
        if self.is_active:
            self.button.config(relief=tk.SUNKEN, text='Mantener centro: ON')
        else:
            self.button.config(relief=tk.RAISED, text='Mantener centro: OFF')


class EventExtractorGUI:
    """Clase principal de la interfaz gráfica."""
    
    def __init__(self):
        self._setup_environment()
        self._init_components()
        self._create_gui()
        self._setup_callbacks()
        self.last_segment = None  # NUEVO: almacenará el último segmento previsualizado
        self._last_window = None  # (start_utc, duration)
    
    def _setup_environment(self):
        """Configura variables de entorno."""
        load_dotenv(find_dotenv())
        self.project_root = os.getenv("PROJECT_LOCAL_ROOT")
        if not self.project_root:
            print("ERROR: PROJECT_LOCAL_ROOT no definido en .env")
            sys.exit(1)
    
    def _init_components(self):
        """Inicializa los componentes principales."""
        self.data_processor = DataProcessor()
        self.time_handler = TimeHandler()
        
    def _create_gui(self):
        """Crea la interfaz gráfica."""
        self.window = tk.Tk()
        self.window.title("Extracción de Eventos - GPD")
        self.window.geometry("800x600")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self._create_file_frame()
        self._create_parameters_frame()
        self._create_actions_frame()
        self._create_plot_frame()
    
    def _create_file_frame(self):
        """Crea el frame para selección de archivo."""
        frame = tk.Frame(self.window)
        frame.pack(fill='x', pady=5)
        
        tk.Label(frame, text="Archivo mseed:").pack(side='left', padx=5)
        self.entry_archivo = tk.Entry(frame, width=50)
        self.entry_archivo.pack(side='left', padx=5)
        
        tk.Button(frame, text="Abrir…", command=self._open_file).pack(side='left', padx=5)
        
        self.lbl_fecha = tk.Label(frame, text="Fecha: --   Inicio: --   Fin: --")
        self.lbl_fecha.pack(side='left', padx=10)
    
    def _create_parameters_frame(self):
        """Crea el frame para parámetros de extracción."""
        frame = tk.Frame(self.window)
        frame.pack(fill='x', pady=5)
        
        # Primera fila
        tk.Label(frame, text="Hora inicio (hh:mm:ss):").grid(row=0, column=0, padx=5, sticky='e')
        self.entry_hora = tk.Entry(frame, width=14)
        self.entry_hora.grid(row=0, column=1, padx=5)
        
        tk.Label(frame, text="Duración (s):").grid(row=0, column=2, padx=5, sticky='e')
        self.spin_duracion = tk.Spinbox(frame, from_=0.1, to=600, increment=0.1, width=6)
        self.spin_duracion.grid(row=0, column=3, padx=5)
        
        # Segunda fila
        tk.Label(frame, text="Desplazamiento (s):").grid(row=1, column=0, padx=5, sticky='e')
        self.entry_shift = tk.Entry(frame, width=6)
        self.entry_shift.insert(0, "0")
        self.entry_shift.grid(row=1, column=1, padx=5, sticky='w')
        
        tk.Label(frame, text="Canal:").grid(row=1, column=2, padx=5, sticky='e')
        
        # Inicializar con valor por defecto hasta que se cargue un archivo
        self.channel_var = tk.StringVar(value="--")
        self.channel_menu = tk.OptionMenu(frame, self.channel_var, "--")
        self.channel_menu.grid(row=1, column=3, padx=5, sticky='w')

    def _update_channel_options(self):
        """Actualiza las opciones de canal basándose en el archivo cargado."""
        if not self.data_processor.stream:
            return
        
        # Obtener todos los canales únicos del stream
        channels = []
        for trace in self.data_processor.stream:
            channel_name = trace.stats.channel
            if channel_name not in channels:
                channels.append(channel_name)
        
        # Actualizar el menú de opciones
        menu = self.channel_menu['menu']
        menu.delete(0, 'end')
        
        # Agregar las nuevas opciones
        for channel in channels:
            menu.add_command(label=channel, command=tk._setit(self.channel_var, channel))
        
        # Establecer el primer canal como predeterminado
        if channels:
            self.channel_var.set(channels[0])
    
    def _create_actions_frame(self):
        """Crea el frame para acciones y controles."""
        frame = tk.Frame(self.window)
        frame.pack(pady=10)
        
        tk.Button(frame, text="Previsualizar", command=self._preview).pack(side='left', padx=10)
        
        tk.Button(frame, text="Guardar mseed", command=self._save_mseed).pack(side='left', padx=10)  # NUEVO
        
        self.btn_keep_center = tk.Button(frame, text="Mantener centro: OFF", command=self._toggle_keep_center)
        self.btn_keep_center.pack(side='left', padx=5)
        
        self.btn_centrar = tk.Button(frame, text="Centrar: OFF", command=self._toggle_centering)
        self.btn_centrar.pack(side='left', padx=10)
        
        tk.Button(frame, text="Salir", command=self._on_close, fg='white', bg='red').pack(side='left', padx=10)
        
        self.lbl_centro = tk.Label(frame, text="Centro: --")
        self.lbl_centro.pack(side='left', padx=10)
        
        self.lbl_pos = tk.Label(frame, text="Posición: --")
        self.lbl_pos.pack(side='left', padx=10)
    
    def _create_plot_frame(self):
        """Crea el frame para el gráfico."""
        self.frame_plot = tk.Frame(self.window)
        self.frame_plot.pack(fill='both', expand=True, pady=5)
        
        self.plot_manager = PlotManager(self.frame_plot)
    
    def _setup_callbacks(self):
        """Configura los callbacks y modos especiales."""
        self.centering_mode = CenteringMode(self.btn_centrar, self.lbl_pos)
        self.keep_center_mode = KeepCenterMode(self.btn_keep_center, self.time_handler)
        
        self.plot_manager.set_callbacks(
            click_callback=self._on_plot_click,
            mouse_move_callback=self._on_mouse_move
        )
        
        # Callback para cambios en duración
        self.spin_duracion.config(command=self._on_duration_change)
    
    def _open_file(self):
        """Abre y procesa un archivo miniSEED."""
        filename = filedialog.askopenfilename(
            title="Selecciona un mseed",
            initialdir=os.path.join(self.project_root, "resultados", "mseed"),
            filetypes=[("MiniSEED", "*.mseed"), ("Todos", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.data_processor.load_file(filename)
            self._update_file_info()
            self._reset_parameters()
            self._update_channel_options()
            
            self.entry_archivo.delete(0, tk.END)
            self.entry_archivo.insert(0, filename)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _update_file_info(self):
        """Actualiza la información del archivo en la interfaz."""
        info = self.data_processor.get_file_info()
        if info:
            self.lbl_fecha.config(
                text=f"Fecha: {info['date']}   Inicio: {info['start_time']}   Fin: {info['end_time']}"
            )
    
    def _reset_parameters(self):
        """Resetea los parámetros a valores por defecto."""
        info = self.data_processor.get_file_info()
        if info:
            self.entry_hora.delete(0, tk.END)
            self.entry_hora.insert(0, info['start_time'])
        
        self.entry_shift.delete(0, tk.END)
        self.entry_shift.insert(0, "0")
        
        # Desactivar modos especiales al cargar nuevo archivo
        self.keep_center_mode.deactivate()
    
    def _get_preview_parameters(self):
        """Obtiene y valida los parámetros para la previsualización."""
        # Validar y obtener hora
        try:
            hora_dt = self.time_handler.parse_time_with_ms(self.entry_hora.get())
        except ValueError as e:
            raise Exception("Formato de hora inicio inválido")
        
        # Validar duración y desplazamiento
        try:
            duration = float(self.spin_duracion.get())
            shift_seconds = float(self.entry_shift.get())
        except ValueError:
            raise Exception("Duración o desplazamiento no válidos")
        
        return {
            'hora_dt': hora_dt,
            'duration': duration,
            'shift_seconds': shift_seconds,
            'channel': self.channel_var.get()
        }
    
    def _preview(self):
        """Previsualiza el segmento seleccionado."""
        if not self.data_processor.stream:
            messagebox.showwarning("Aviso", "Primero abre un archivo mseed.")
            return
        
        try:
            params = self._get_preview_parameters()
            
            # Si está activo "Mantener centro", usar la hora recalculada
            if self.keep_center_mode.is_active:
                new_start_time = self.keep_center_mode.calculate_new_start_time(params['duration'])
                if new_start_time:
                    params['hora_dt'] = new_start_time
                    params['shift_seconds'] = 0  # Sin desplazamiento
            
            # Crear tiempo UTC con desplazamiento
            start_utc, start_dt = self.time_handler.create_utc_datetime(
                self.data_processor.file_starttime.date,
                params['hora_dt'],
                params['shift_seconds']
            )
            
            # Extraer segmento
            segment = self.data_processor.extract_segment(
                start_utc, params['duration'], params['channel']
            )

            self.last_segment = segment  # NUEVO: guardar el ultimo segmento previsualizado
            
            # Calcular el centro actual
            center_utc = start_utc + (params['duration'] / 2.0)
            
            # Actualizar interfaz
            self._update_time_display(start_dt)
            self._update_center_display(start_utc, params['duration'])
            
            # SIEMPRE actualizar la referencia del centro después de cada previsualización
            # Esto asegura que "Mantener centro" use siempre el centro más reciente
            self.keep_center_mode.set_center_reference(
                center_utc, 
                self.data_processor.file_starttime.date
            )
            
            self.centering_mode.deactivate()
            
            # Crear gráfico
            self.plot_manager.create_plot(
                segment, params['duration'], self.entry_archivo.get()
            )

            # Guardar la ultima ventana usada para previsualizacion
            self._last_window = (start_utc, params['duration'])
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _update_time_display(self, start_dt):
        """Actualiza la visualización del tiempo y resetea el desplazamiento."""
        new_time_str = self.time_handler.format_time_with_ms(start_dt)
        self.entry_hora.delete(0, tk.END)
        self.entry_hora.insert(0, new_time_str)
        
        self.entry_shift.delete(0, tk.END)
        self.entry_shift.insert(0, "0")
    
    def _update_center_display(self, start_utc, duration):
        """Actualiza la visualización del centro temporal."""
        center_utc = start_utc + (duration / 2.0)
        center_dt = center_utc.datetime
        center_str = self.time_handler.format_time_with_ms(center_dt)
        self.lbl_centro.config(text=f"Centro: {center_str}")
    
    def _toggle_centering(self):
        """Alterna el modo de centrado."""
        self.centering_mode.toggle()
    
    def _toggle_keep_center(self):
        """Alterna el modo mantener centro."""
        self.keep_center_mode.toggle()
    
    def _on_duration_change(self):
        """Maneja cambios en la duración cuando está activo 'Mantener centro'."""
        if not self.keep_center_mode.is_active:
            return
        
        try:
            new_duration = float(self.spin_duracion.get())
            new_start_time = self.keep_center_mode.calculate_new_start_time(new_duration)
            
            if new_start_time:
                # Actualizar el campo de hora inicio
                time_str = self.time_handler.format_time_with_ms(
                    datetime.combine(datetime.min.date(), new_start_time)
                )
                self.entry_hora.delete(0, tk.END)
                self.entry_hora.insert(0, time_str)
                
                # Resetear desplazamiento
                self.entry_shift.delete(0, tk.END)
                self.entry_shift.insert(0, "0")
                
        except ValueError:
            pass  # Ignorar valores inválidos durante la escritura
    
    def _on_plot_click(self, event):
        """Maneja clics en el gráfico para centrado."""
        if not self.centering_mode.is_active or event.inaxes is None:
            return
        
        x_click = event.xdata
        duration = float(self.spin_duracion.get())
        center = duration / 2.0
        delta = x_click - center
        
        self.entry_shift.delete(0, tk.END)
        self.entry_shift.insert(0, f"{delta:.2f}")
        
        self.centering_mode.toggle()
    
    def _on_mouse_move(self, event):
        """Maneja el movimiento del mouse sobre el gráfico."""
        if not event.inaxes or event.xdata is None:
            if self.centering_mode.is_active:
                self.lbl_pos.config(text="Δ: --")
            else:
                self.lbl_pos.config(text="Posición: --")
            return
        
        if self.centering_mode.is_active:
            duration = float(self.spin_duracion.get())
            center = duration / 2.0
            delta = event.xdata - center
            self.lbl_pos.config(text=f"Δ: {delta:+.2f} s")
        else:
            self.lbl_pos.config(text=f"Posición: {event.xdata:.2f} s")
    
    def _save_mseed(self):
        """
        Guarda el evento en miniSEED incluyendo **todos los canales** presentes en la ventana
        seleccionada (no solo el canal previsualizado).
        """
        
        # Comprobaciones basicas
        if self.data_processor is None or self.data_processor.stream is None:
            messagebox.showerror("Error", "No hay archivo cargado.")
            return

        # Elegir ruta de salida
        filename = filedialog.asksaveasfilename(
            title="Guardar evento miniSEED",
            defaultextension=".mseed",
            filetypes=[("miniSEED", ".mseed"), ("Todos", ".*")],
        )
        if not filename:
            return

        try:
            # 1) Determinar la ventana a guardar (preferir la usada en previsualizacion)
            window = getattr(self, "_last_window", None)
            if window is None:
                # Fallback: reconstruir desde los widgets si no hubo previsualizacion
                params = self._gather_params()  # Debe existir en tu clase
                start_utc, _ = self.time_handler.create_utc_datetime(
                    self.data_processor.file_starttime.date,
                    params['hora_dt'],
                    params['shift_seconds']
                )
                duration = params['duration']
            else:
                start_utc, duration = window

            # 2) Extraer **todos** los canales en esa ventana (nuevo metodo en DataProcessor)
            segmento_a_guardar = self.data_processor.extract_segment_multi(start_utc, duration)  # NUEVO
            if len(segmento_a_guardar) == 0:
                messagebox.showerror("Error", "No hay datos en la ventana seleccionada para ningun canal.")
                return

            # (Opcional) Avisar si faltan canales esperados ENT/ENR/ENV
            # esperados = {"ENT", "ENR", "ENV"}
            # presentes = {tr.stats.channel for tr in segmento_a_guardar}
            # faltan = sorted(esperados - presentes)
            # if faltan:
            #     messagebox.showwarning("Aviso", "Se guardo el evento, pero faltaron canales: " + ", ".join(faltan))

            # 3) Elegir codificacion segun dtype (robusto)
            # Detectar si algun trace es float; si mezcla tipos, homogenizar a float32
            any_float = any(np.issubdtype(tr.data.dtype, np.floating) for tr in segmento_a_guardar)
            if any_float:
                for tr in segmento_a_guardar:
                    tr.data = tr.data.astype(np.float32, copy=False)
                encoding = 4   # FLOAT32
            else:
                # Enteros -> STEIM2 asegurando int32
                for tr in segmento_a_guardar:
                    if tr.data.dtype != np.int32:
                        tr.data = tr.data.astype(np.int32, copy=False)
                encoding = 11  # STEIM2

            # 4) Escribir archivo miniSEED con todos los trazos
            segmento_a_guardar.write(filename, format="MSEED", encoding=encoding, reclen=4096)
            
            messagebox.showinfo("Exito", f"Segmento guardado en:\n{os.path.abspath(filename)}")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar: {e}")
    
    
    
    def _on_close(self):
        """Maneja el cierre de la aplicación."""
        self.window.quit()
        self.window.destroy()
        sys.exit(0)
    
    def run(self):
        """Ejecuta la aplicación."""
        self.window.mainloop()


def main():
    """Función principal."""
    app = EventExtractorGUI()
    app.run()


if __name__ == "__main__":
    main()
