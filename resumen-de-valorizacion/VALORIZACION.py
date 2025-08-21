import subprocess
import sys
import os
import concurrent.futures
import threading
import time
from datetime import datetime, timedelta

def install_dependencies():
    """
    Check for and install required dependencies
    """
    required_packages = {
        'pandas': 'pandas>=2.0.0',
        'numpy': 'numpy>=1.20.0',
        'xlrd': 'xlrd>=2.0.1',
        'xlsxwriter': 'xlsxwriter>=3.0.0',
        'openpyxl': 'openpyxl>=3.0.0'
    }
    
    missing_packages = []
    
    # Check each package
    for package_name, package_spec in required_packages.items():
        try:
            __import__(package_name)
            print(f"✅ {package_name} ya está instalado")
        except ImportError:
            print(f"❌ {package_name} no está disponible")
            missing_packages.append(package_spec)
    
    # Install missing packages
    if missing_packages:
        print(f"\n🔄 Instalando paquetes faltantes: {', '.join(missing_packages)}")
        try:
            for package in missing_packages:
                print(f"Instalando {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("✅ ¡Todas las dependencias se instalaron correctamente!")
            print("🔄 Reiniciando script para usar las nuevas dependencias...\n")
            
            # Restart the script to use newly installed packages
            python = sys.executable
            os.execl(python, python, *sys.argv)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando paquetes: {e}")
            print("Por favor instale los paquetes manualmente usando:")
            print(f"py -m pip install {' '.join(missing_packages)}")
            input("Presione Enter para continuar de todos modos...")
    
    print("📦 ¡Todas las dependencias están listas!\n")


# Install dependencies first
install_dependencies()

# Now import the packages
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

def split_datetime_column(df, datetime_col_index=None):
    """
    Optimized datetime column splitting with vectorized operations
    """
    if datetime_col_index is None:
        # Find the first column that contains datetime-like data
        for i, col in enumerate(df.columns):
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'time' in col.lower() or 'date' in col.lower():
                datetime_col_index = i
                break
    
    if datetime_col_index is not None:
        datetime_col = df.columns[datetime_col_index]
        
        # Vectorized datetime conversion with error handling
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        
        # Extract time data using vectorized operations
        time_data = df[datetime_col].dt.time
        
        # Convert to date using vectorized operations
        df[datetime_col] = df[datetime_col].dt.date
        
        # Insert time column
        time_col_name = f"{datetime_col}_time"
        df.insert(datetime_col_index + 1, time_col_name, time_data)
    
    return df

def remove_closest_readings_fast(timestamps, indices, max_readings):
    """
    Optimized selection-based algorithm for maximum temporal distribution
    Returns: (indices_to_keep, indices_deleted)
    """
    if len(indices) <= max_readings:
        return indices.copy(), []
    
    # Convert to numpy arrays for efficiency
    ts = np.array(timestamps)
    idx = np.array(indices)
    
    # Strategy: Select readings with best temporal distribution
    # Use equal time intervals to maximize spread
    
    # Always keep first and last readings for boundary coverage
    if max_readings >= 2:
        selected_positions = [0, len(ts) - 1]
        remaining_slots = max_readings - 2
    else:
        # If only 1 reading needed, pick the middle one
        middle_pos = len(ts) // 2
        selected_positions = [middle_pos]
        remaining_slots = 0
    
    # Fill remaining slots with evenly distributed readings
    if remaining_slots > 0:
        # Calculate interval size
        time_span = ts[-1] - ts[0]
        
        # Fix: Convert numpy timedelta64 to seconds for comparison
        if hasattr(time_span, 'astype'):
            # For numpy timedelta64 objects
            time_span_seconds = time_span.astype('timedelta64[s]').astype(float)
        else:
            # For pandas/python timedelta objects
            time_span_seconds = time_span.total_seconds()
        
        if time_span_seconds > 0:  # Avoid division by zero
            # Target positions in time
            for i in range(1, remaining_slots + 1):
                # Calculate target time for this slot
                target_fraction = i / (remaining_slots + 1)
                target_time = ts[0] + target_fraction * time_span
                
                # Find closest reading to target time (excluding already selected)
                available_mask = np.ones(len(ts), dtype=bool)
                available_mask[selected_positions] = False
                
                if np.any(available_mask):
                    available_indices = np.where(available_mask)[0]
                    time_distances = np.abs(ts[available_indices] - target_time)
                    best_relative_idx = np.argmin(time_distances)
                    best_absolute_idx = available_indices[best_relative_idx]
                    selected_positions.append(best_absolute_idx)
        else:
            # All timestamps are the same - just pick evenly spaced indices
            step = len(ts) // (remaining_slots + 1)
            for i in range(1, remaining_slots + 1):
                pos = min(i * step, len(ts) - 2)  # Avoid duplicating last position
                if pos not in selected_positions:
                    selected_positions.append(pos)
    
    # Sort positions to maintain chronological order
    selected_positions = sorted(selected_positions)
    
    # Ensure we don't exceed max_readings due to any logic errors
    selected_positions = selected_positions[:max_readings]
    
    # Create boolean mask for selection
    keep_mask = np.zeros(len(idx), dtype=bool)
    keep_mask[selected_positions] = True
    
    # Calculate results
    indices_to_keep = idx[keep_mask]
    indices_deleted = idx[~keep_mask]
    
    return indices_to_keep, indices_deleted

def limit_readings_per_day_optimized(df, max_readings=7):
    """
    Highly optimized reading limitation using vectorized operations, parallel processing and efficient algorithms
    Returns: (limited_df, deleted_df)
    """
    # Find columns using vectorized string operations
    date_col = time_col = meter_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'record time' in col_lower and 'time_time' not in col_lower:
            date_col = col
        elif 'record time_time' in col_lower or 'time_time' in col_lower:
            time_col = col
        elif 'meter id' in col_lower or 'meter_id' in col_lower:
            meter_col = col

    if not all([date_col, time_col, meter_col]):
        print("No se pudieron encontrar las columnas requeridas (fecha, hora, ID del medidor)")
        print("Columnas disponibles:", list(df.columns))
        return df, pd.DataFrame()

    # Work with copy to avoid warnings
    df = df.copy()
    
    # Vectorized datetime creation
    df['temp_datetime'] = pd.to_datetime(
        df[date_col].astype(str) + ' ' + df[time_col].astype(str), 
        errors='coerce'
    )
    
    # Single sort operation
    df = df.sort_values([meter_col, date_col, 'temp_datetime']).reset_index(drop=True)
    
    # Efficient grouping
    grouped = df.groupby([meter_col, date_col])
    group_sizes = grouped.size()
    
    # Pre-filter groups that need processing
    groups_to_process = group_sizes[group_sizes > max_readings].index
    groups_no_process = group_sizes[group_sizes <= max_readings].index
    
    # Early return if no processing needed
    if len(groups_to_process) == 0:
        return df.drop('temp_datetime', axis=1), pd.DataFrame()
    
    # Use parallel processing for large datasets
    if len(groups_to_process) > 10:
        return process_groups_parallel(df, grouped, groups_to_process, groups_no_process, max_readings)
    else:
        return process_groups_sequential(df, grouped, groups_to_process, groups_no_process, max_readings)

def process_groups_parallel(df, grouped, groups_to_process, groups_no_process, max_readings):
    """
    Process groups in parallel for better performance on large datasets
    """
    
    # Split groups into chunks for parallel processing
    groups_list = list(groups_to_process)
    chunk_size = max(1, len(groups_list) // min(4, len(groups_list)))
    chunks = [groups_list[i:i + chunk_size] for i in range(0, len(groups_list), chunk_size)]
    
    def process_chunk(chunk_groups):
        """Process a chunk of groups"""
        chunk_keep_indices = []
        chunk_deleted_indices = []
        
        for (meter_id, date) in chunk_groups:
            group = grouped.get_group((meter_id, date))
            original_count = len(group)
            
            # Convert to numpy arrays for fast operations
            timestamps = group['temp_datetime'].values
            indices = group.index.values
            
            # Use optimized algorithm
            indices_to_keep, indices_deleted = remove_closest_readings_fast(timestamps, indices, max_readings)
            chunk_keep_indices.extend(indices_to_keep)
            chunk_deleted_indices.extend(indices_deleted)
        
        return chunk_keep_indices, chunk_deleted_indices
    
    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        
        keep_indices = []
        deleted_indices = []
        
        for future in concurrent.futures.as_completed(futures):
            try:
                chunk_keep, chunk_deleted = future.result()
                keep_indices.extend(chunk_keep)
                deleted_indices.extend(chunk_deleted)
            except Exception as e:
                pass
    
    # Add indices from groups that don't need processing
    for (meter_id, date) in groups_no_process:
        group = grouped.get_group((meter_id, date))
        keep_indices.extend(group.index)
    
    # Create result DataFrames
    result_df = df.loc[keep_indices].sort_index()
    result_df = result_df.drop('temp_datetime', axis=1)
    
    # Create deleted DataFrame
    if deleted_indices:
        deleted_df = df.loc[deleted_indices].sort_index()
        deleted_df = deleted_df.drop('temp_datetime', axis=1)
    else:
        deleted_df = pd.DataFrame()
    
    return result_df, deleted_df

def process_groups_sequential(df, grouped, groups_to_process, groups_no_process, max_readings):
    """
    Process groups sequentially for smaller datasets
    """
    # Pre-allocate lists for better memory management
    keep_indices = []
    deleted_indices = []
    
    # Process only groups that need reduction
    for (meter_id, date) in groups_to_process:
        group = grouped.get_group((meter_id, date))
        original_count = len(group)
        
        # Convert to numpy arrays for fast operations
        timestamps = group['temp_datetime'].values
        indices = group.index.values
        
        # Use optimized algorithm
        indices_to_keep, indices_deleted = remove_closest_readings_fast(timestamps, indices, max_readings)
        keep_indices.extend(indices_to_keep)
        deleted_indices.extend(indices_deleted)
    
    # Add indices from groups that don't need processing
    for (meter_id, date) in groups_no_process:
        group = grouped.get_group((meter_id, date))
        keep_indices.extend(group.index)
    
    # Create result DataFrames
    result_df = df.loc[keep_indices].sort_index()
    result_df = result_df.drop('temp_datetime', axis=1)
    
    # Create deleted DataFrame
    if deleted_indices:
        deleted_df = df.loc[deleted_indices].sort_index()
        deleted_df = deleted_df.drop('temp_datetime', axis=1)
    else:
        deleted_df = pd.DataFrame()
    
    return result_df, deleted_df

def split_by_gateway_optimized(df):
    """
    Optimized gateway splitting using vectorized operations
    """
    # Find Gateway column efficiently
    gateway_col = next((col for col in df.columns if 'gateway' in col.lower()), None)
    
    if gateway_col is None:
        print("¡No se encontró columna Gateway!")
        return df, pd.DataFrame()
    
    # Vectorized boolean operations for splitting
    has_gateway_mask = df[gateway_col].notna() & (df[gateway_col] != '')
    df_with_gateway = df[has_gateway_mask].copy()
    df_without_gateway = df[~has_gateway_mask].copy()
    
    return df_with_gateway, df_without_gateway

def filter_walkby_latest_per_day(df_walkby):
    """
    Filter WALKBY readings to keep only the latest reading per meter per day
    If multiple WALKBY readings exist for the same meter on the same day, keep only the latest one
    Returns: (filtered_df, deleted_df)
    """
    if df_walkby.empty:
        return df_walkby, pd.DataFrame()
    
    # Find required columns
    meter_col = date_col = time_col = None
    
    for col in df_walkby.columns:
        col_lower = col.lower()
        if 'meter id' in col_lower or 'meter_id' in col_lower:
            meter_col = col
        elif 'record time' in col_lower and 'time_time' not in col_lower:
            date_col = col
        elif 'record time_time' in col_lower or 'time_time' in col_lower:
            time_col = col
    
    if not all([meter_col, date_col, time_col]):
        print("⚠️ No se pudieron encontrar las columnas requeridas para filtrar WALKBY")
        return df_walkby, pd.DataFrame()
    
    # Work with copy
    df = df_walkby.copy()
    
    # Create datetime column for proper sorting
    df['temp_datetime'] = pd.to_datetime(
        df[date_col].astype(str) + ' ' + df[time_col].astype(str), 
        errors='coerce'
    )
    
    # Sort by meter, date, and time (latest first)
    df = df.sort_values([meter_col, date_col, 'temp_datetime'], ascending=[True, True, False]).reset_index(drop=True)
    
    # Group by meter and date to find duplicates
    grouped = df.groupby([meter_col, date_col])
    group_sizes = grouped.size()
    
    # Find groups with multiple readings per day
    groups_with_duplicates = group_sizes[group_sizes > 1].index
    
    if len(groups_with_duplicates) == 0:
        return df_walkby, pd.DataFrame()
    
    # Collect indices to keep (latest reading per day) and to delete
    keep_indices = []
    deleted_indices = []
    
    for (meter_id, date) in groups_with_duplicates:
        group = grouped.get_group((meter_id, date))
        
        # Keep the first row (latest time due to descending sort)
        latest_idx = group.index[0]
        keep_indices.append(latest_idx)
        
        # Mark the rest for deletion
        duplicate_indices = group.index[1:].tolist()
        deleted_indices.extend(duplicate_indices)
    
    # Add indices from groups without duplicates
    groups_without_duplicates = group_sizes[group_sizes == 1].index
    for (meter_id, date) in groups_without_duplicates:
        group = grouped.get_group((meter_id, date))
        keep_indices.extend(group.index)
    
    # Create filtered DataFrame
    filtered_df = df.loc[keep_indices].sort_index()
    filtered_df = filtered_df.drop('temp_datetime', axis=1)
    
    # Create deleted DataFrame
    if deleted_indices:
        deleted_df = df.loc[deleted_indices].sort_index()
        deleted_df = deleted_df.drop('temp_datetime', axis=1)
        # Add deletion reason
        deleted_df['Motivo_Eliminacion'] = 'Lectura WALKBY duplicada'
    else:
        deleted_df = pd.DataFrame()
    
    return filtered_df, deleted_df

def limit_pure_walkby_to_first_per_month(df_walkby, df_gateway):
    """
    For pure WALKBY meters (meters that don't exist in gateway data), keep only the first reading per month.
    All subsequent readings in the same month are deleted.
    Returns: (filtered_df, deleted_df)
    """
    if df_walkby.empty:
        return df_walkby, pd.DataFrame()
    
    # Find required columns
    meter_col = date_col = time_col = None
    
    for col in df_walkby.columns:
        col_lower = col.lower()
        if 'meter id' in col_lower or 'meter_id' in col_lower:
            meter_col = col
        elif 'record time' in col_lower and 'time_time' not in col_lower:
            date_col = col
        elif 'record time_time' in col_lower or 'time_time' in col_lower:
            time_col = col
    
    if not all([meter_col, date_col, time_col]):
        print("⚠️ No se pudieron encontrar las columnas requeridas para filtrar WALKBY por mes")
        return df_walkby, pd.DataFrame()
    
    # Find pure WALKBY meter IDs (meters that don't exist in gateway data)
    pure_walkby_meters = set()
    if not df_walkby.empty and meter_col in df_walkby.columns:
        walkby_meter_ids = set(df_walkby[meter_col].unique())
        
        if not df_gateway.empty and meter_col in df_gateway.columns:
            gateway_meter_ids = set(df_gateway[meter_col].unique())
            pure_walkby_meters = walkby_meter_ids - gateway_meter_ids
        else:
            pure_walkby_meters = walkby_meter_ids
    
    if not pure_walkby_meters:
        return df_walkby, pd.DataFrame()
    
    # Work with copy
    df = df_walkby.copy()
    
    # Create datetime column for proper sorting
    df['temp_datetime'] = pd.to_datetime(
        df[date_col].astype(str) + ' ' + df[time_col].astype(str), 
        errors='coerce'
    )
    
    # Add year-month column for grouping
    df['year_month'] = df['temp_datetime'].dt.to_period('M')
    
    # Sort by meter, year-month, and datetime (earliest first)
    df = df.sort_values([meter_col, 'year_month', 'temp_datetime'], ascending=[True, True, True]).reset_index(drop=True)
    
    # Filter only pure walkby meters
    pure_walkby_mask = df[meter_col].isin(pure_walkby_meters)
    df_pure_walkby = df[pure_walkby_mask].copy()
    df_mixed_walkby = df[~pure_walkby_mask].copy()
    
    if df_pure_walkby.empty:
        # No pure walkby meters to process
        result_df = df.drop(['temp_datetime', 'year_month'], axis=1)
        return result_df, pd.DataFrame()
    
    # Group by meter and year-month to find multiple readings per month
    grouped = df_pure_walkby.groupby([meter_col, 'year_month'])
    group_sizes = grouped.size()
    
    # Find groups with multiple readings per month
    groups_with_multiple = group_sizes[group_sizes > 1].index
    
    if len(groups_with_multiple) == 0:
        # No pure walkby meters have multiple readings per month
        result_df = df.drop(['temp_datetime', 'year_month'], axis=1)
        return result_df, pd.DataFrame()
    
    # Collect indices to keep (first reading per month) and to delete
    keep_indices = []
    deleted_indices = []
    
    for (meter_id, year_month) in groups_with_multiple:
        group = grouped.get_group((meter_id, year_month))
        
        # Keep the first row (earliest time due to ascending sort)
        first_idx = group.index[0]
        keep_indices.append(first_idx)
        
        # Mark the rest for deletion
        subsequent_indices = group.index[1:].tolist()
        deleted_indices.extend(subsequent_indices)
        
        print(f"📅 Medidor WALKBY puro {meter_id} en {year_month}: Manteniendo primera lectura, eliminando {len(subsequent_indices)} lecturas adicionales")
    
    # Add indices from groups without multiple readings per month
    groups_without_multiple = group_sizes[group_sizes == 1].index
    for (meter_id, year_month) in groups_without_multiple:
        group = grouped.get_group((meter_id, year_month))
        keep_indices.extend(group.index)
    
    # Add all mixed walkby meters (not pure walkby)
    if not df_mixed_walkby.empty:
        keep_indices.extend(df_mixed_walkby.index)
    
    # Create filtered DataFrame
    filtered_df = df.loc[keep_indices].sort_index()
    filtered_df = filtered_df.drop(['temp_datetime', 'year_month'], axis=1)
    
    # Create deleted DataFrame
    if deleted_indices:
        deleted_df = df.loc[deleted_indices].sort_index()
        deleted_df = deleted_df.drop(['temp_datetime', 'year_month'], axis=1)
        # Add deletion reason
        deleted_df['Motivo_Eliminacion'] = 'Lectura WALKBY adicional en el mes (solo primera lectura permitida)'
    else:
        deleted_df = pd.DataFrame()
    
    total_deleted = len(deleted_indices)
    if total_deleted > 0:
        print(f"✅ Filtrado de WALKBY puros completado: {total_deleted} lecturas adicionales eliminadas")
    
    return filtered_df, deleted_df

def get_file_paths():
    """
    Get file paths through drag-and-drop GUI or manual input
    Returns: (data_file_path, meters_file_path, incidents_file_path)
    """
    # Check if files were dragged onto script
    if len(sys.argv) > 1:
        file_paths = sys.argv[1:]
        print("=== Procesador Optimizado de Datos de Medidores ===")
        print("🎯 MODO ARRASTRAR Y SOLTAR ACTIVADO")
        print("=" * 52)
        
        valid_files = []
        for file_path in file_paths:
            if os.path.exists(file_path) and file_path.lower().endswith(('.xls', '.xlsx')):
                valid_files.append(file_path)
        
        if len(valid_files) >= 1:
            data_file = valid_files[0]
            meters_file = valid_files[1] if len(valid_files) >= 2 else None
            incidents_file = valid_files[2] if len(valid_files) >= 3 else None
            
            print(f"✅ Archivo de datos: {os.path.basename(data_file)}")
            if meters_file:
                print(f"✅ Archivo de medidores: {os.path.basename(meters_file)}")
            else:
                print("ℹ️ Sin archivo de medidores")
            if incidents_file:
                print(f"✅ Archivo de incidencias: {os.path.basename(incidents_file)}")
            else:
                print("ℹ️ Sin archivo de incidencias")
            
            return data_file, meters_file, incidents_file
        else:
            print("❌ No se encontraron archivos válidos de Excel")
            print("Cambiando a modo interactivo...")
            print()
    
    print("=== Procesador Optimizado de Datos de Medidores ===")
    print("📋 MODO INTERACTIVO")
    print("=" * 52)
    print("Necesitas seleccionar:")
    print("1. Archivo de DATOS (obligatorio) - contiene las lecturas de medidores")
    print("2. Archivo de MEDIDORES (opcional) - contiene lista completa de medidores del sistema")
    print("3. Archivo de INCIDENCIAS (opcional) - contiene medidores con incidencias")
    print()
    print("💡 CONSEJO: Arrastra los tres archivos sobre VALORIZACION.PY (datos, medidores, incidencias)")
    print()
    
    # Get data file
    print("📊 SELECCIONAR ARCHIVO DE DATOS:")
    choice = input("1. Explorador de archivos | 2. Escribir ruta: ").strip()
    
    data_file = None
    if choice == "1":
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            data_file = filedialog.askopenfilename(
                title="Seleccionar Archivo de DATOS (lecturas)",
                filetypes=[("Archivos de Excel", "*.xlsx *.xls"), ("Todos los archivos", "*.*")]
            )
            root.destroy()
            
            if not data_file:
                print("❌ No se seleccionó archivo de datos")
                return None, None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None
            
    elif choice == "2":
        print("Pega la ruta del archivo de datos:")
        data_file = input().strip().strip('"').strip("'")
        if not data_file:
            print("❌ No se proporcionó archivo de datos")
            return None, None
    else:
        print("❌ Opción inválida")
        return None, None
    
    print(f"✅ Archivo de datos: {os.path.basename(data_file)}")
    
    # Get meters file (optional)
    print("\n🔢 SELECCIONAR ARCHIVO DE MEDIDORES (opcional):")
    print("Este archivo debe contener la lista completa de medidores del sistema")
    choice = input("1. Explorador | 2. Escribir ruta | 3. Omitir: ").strip()
    
    meters_file = None
    if choice == "1":
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            meters_file = filedialog.askopenfilename(
                title="Seleccionar Archivo de MEDIDORES (lista completa)",
                filetypes=[("Archivos de Excel", "*.xlsx *.xls"), ("Todos los archivos", "*.*")]
            )
            root.destroy()
            
        except Exception as e:
            print(f"⚠️ Error seleccionando archivo de medidores: {e}")
            
    elif choice == "2":
        print("Pega la ruta del archivo de medidores:")
        meters_file = input().strip().strip('"').strip("'")
        if not meters_file:
            meters_file = None
            
    elif choice == "3":
        meters_file = None
    else:
        print("⚠️ Opción inválida, omitiendo archivo de medidores")
        meters_file = None
    
    if meters_file:
        print(f"✅ Archivo de medidores: {os.path.basename(meters_file)}")
    else:
        print("ℹ️ Sin archivo de medidores - solo se procesarán medidores con lecturas")
    
    # Get incidents file (optional)
    print("\n🚨 SELECCIONAR ARCHIVO DE INCIDENCIAS (opcional):")
    print("Este archivo debe contener medidores con incidencias (columna 1 = MeterID, columna 5 = tipo incidencia)")
    choice = input("1. Explorador | 2. Escribir ruta | 3. Omitir: ").strip()
    
    incidents_file = None
    if choice == "1":
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            incidents_file = filedialog.askopenfilename(
                title="Seleccionar Archivo de INCIDENCIAS",
                filetypes=[("Archivos de Excel", "*.xlsx *.xls"), ("Todos los archivos", "*.*")]
            )
            root.destroy()
            
        except Exception as e:
            print(f"⚠️ Error seleccionando archivo de incidencias: {e}")
            
    elif choice == "2":
        print("Pega la ruta del archivo de incidencias:")
        incidents_file = input().strip().strip('"').strip("'")
        if not incidents_file:
            incidents_file = None
            
    elif choice == "3":
        incidents_file = None
    else:
        print("⚠️ Opción inválida, omitiendo archivo de incidencias")
        incidents_file = None
    
    if incidents_file:
        print(f"✅ Archivo de incidencias: {os.path.basename(incidents_file)}")
    else:
        print("ℹ️ Sin archivo de incidencias - no se agregarán datos de incidencias")
    
    return data_file, meters_file, incidents_file

def read_meters_file(meters_file_path):
    """
    Read the meters file and extract the list of all meters in the system
    Returns a tuple: (set of meter IDs, customer mapping dict)
    """
    if not meters_file_path or not os.path.exists(meters_file_path):
        return set(), {}
    
    try:
        print(f"📋 Leyendo archivo de medidores: {os.path.basename(meters_file_path)}")
        
        # Read the file
        if meters_file_path.endswith('.xls'):
            all_sheets = pd.read_excel(meters_file_path, sheet_name=None, engine='xlrd')
        else:
            all_sheets = pd.read_excel(meters_file_path, sheet_name=None, engine='openpyxl')
        
        # Combine all sheets
        dataframes = [df for df in all_sheets.values() if not df.empty]
        if not dataframes:
            print("⚠️ No se encontraron datos en el archivo de medidores")
            return set(), {}
        
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        
        # Find meter ID column - prioritize exact matches for "MeterID" format
        meter_col = None
        customer_col = None
        
        # First check for exact matches (case-sensitive for MeterID format)
        exact_matches = ['MeterID', 'Meter ID', 'meter_id', 'ID_MEDIDOR', 'medidor']
        for exact_col in exact_matches:
            if exact_col in combined_df.columns:
                meter_col = exact_col
                break
        
        # If no exact match, use case-insensitive search
        if meter_col is None:
            for col in combined_df.columns:
                col_lower = col.lower()
                if 'meter id' in col_lower or 'meter_id' in col_lower or 'medidor' in col_lower:
                    meter_col = col
                    break
        
        if meter_col is None:
            print("⚠️ No se encontró columna de ID de medidor en archivo de medidores")
            print("Columnas disponibles:", list(combined_df.columns))
            return set(), {}
        
        # Find customer/NIS column - look for common customer column names
        customer_matches = ['NIS', 'Customer ID', 'CustomerID', 'customer_id', 'Cliente', 'CLIENTE', 'ID_CLIENTE']
        for exact_col in customer_matches:
            if exact_col in combined_df.columns:
                customer_col = exact_col
                break
        
        # If no exact match, use case-insensitive search
        if customer_col is None:
            for col in combined_df.columns:
                col_lower = col.lower()
                if ('nis' in col_lower or 'customer' in col_lower or 'cliente' in col_lower or 
                    'client' in col_lower) and 'id' in col_lower:
                    customer_col = col
                    break
        
        # If still no customer column found, check if NIS is in column index 1 (common pattern)
        if customer_col is None and len(combined_df.columns) > 1:
            # Try second column as NIS/Customer ID
            customer_col = combined_df.columns[1]
        
        # Extract meter IDs and customer mapping
        meter_ids = set()
        customer_mapping = {}
        
        # Process data starting from row 2 (index 1) - first data row after headers
        for idx in range(1, len(combined_df)):
            try:
                meter_id = combined_df.iloc[idx, combined_df.columns.get_loc(meter_col)]
                
                # Get customer ID if customer column exists
                customer_id = None
                if customer_col:
                    try:
                        customer_id = combined_df.iloc[idx, combined_df.columns.get_loc(customer_col)]
                    except:
                        customer_id = None
                
                # Skip if meter ID is null/empty
                if pd.isna(meter_id):
                    continue
                
                # Convert to string and clean
                meter_id_str = str(meter_id).strip()
                
                if meter_id_str:
                    meter_ids.add(meter_id_str)
                    
                    # Add customer mapping if available
                    if customer_id is not None and not pd.isna(customer_id):
                        customer_id_str = str(customer_id).strip()
                        if customer_id_str:
                            customer_mapping[meter_id_str] = customer_id_str
                    else:
                        customer_mapping[meter_id_str] = 'SIN CLIENTE'
                    
            except (IndexError, ValueError, KeyError) as e:
                # Skip rows with missing data
                continue
        
        print(f"✅ Archivo de medidores procesado: {len(meter_ids)} medidores encontrados")
        
        return meter_ids, customer_mapping
        
    except Exception as e:
        print(f"❌ Error leyendo archivo de medidores: {str(e)}")
        return set(), {}

def read_incidents_file(incidents_file_path):
    """
    Read the incidents file and extract meter incidents data
    New format for testval.py: Two columns with headers
    - Column 1: "ID MEDIDOR" (Meter ID)  
    - Column 2: "INCIDENCIA/SIN LECTURA" (Incident Type)
    Returns a dictionary: {meter_id: incident_type}
    """
    if not incidents_file_path or not os.path.exists(incidents_file_path):
        return {}
    
    try:
        print(f"🚨 Leyendo archivo de incidencias (formato testval): {os.path.basename(incidents_file_path)}")
        
        # Read the file
        if incidents_file_path.endswith('.xls'):
            all_sheets = pd.read_excel(incidents_file_path, sheet_name=None, engine='xlrd')
        else:
            all_sheets = pd.read_excel(incidents_file_path, sheet_name=None, engine='openpyxl')
        
        # Use the first sheet
        first_sheet = next(iter(all_sheets.values()))
        
        if first_sheet.empty:
            print("⚠️ No se encontraron datos en el archivo de incidencias")
            return {}
        
        # Find the meter ID and incident columns by header names
        meter_col = None
        incident_col = None
        
        # Search for meter ID column
        for col in first_sheet.columns:
            col_str = str(col).upper().strip()
            if 'ID MEDIDOR' in col_str or 'MEDIDOR' in col_str or 'METER' in col_str:
                meter_col = col
                break
        
        # Search for incident column
        for col in first_sheet.columns:
            col_str = str(col).upper().strip()
            if ('INCIDENCIA' in col_str and 'SIN LECTURA' in col_str) or \
               'INCIDENCIA/SIN LECTURA' in col_str or \
               ('INCIDENCIA' in col_str or 'SIN LECTURA' in col_str):
                incident_col = col
                break
        
        # If columns not found by name, use positional approach
        if meter_col is None:
            if len(first_sheet.columns) >= 1:
                meter_col = first_sheet.columns[0]  # First column
            else:
                print("❌ No se encontró columna de medidor")
                return {}
        
        if incident_col is None:
            if len(first_sheet.columns) >= 2:
                incident_col = first_sheet.columns[1]  # Second column
            else:
                print("❌ No se encontró columna de incidencia")
                return {}
        
        # Extract incidents data
        incidents_dict = {}
        
        # Process data starting from row 1 (index 0 since we have headers)
        for idx in range(len(first_sheet)):
            try:
                # Get meter ID from meter column
                meter_id = first_sheet.iloc[idx][meter_col]
                # Get incident type from incident column
                incident_type = first_sheet.iloc[idx][incident_col]
                
                # Skip if either value is null/empty
                if pd.isna(meter_id) or pd.isna(incident_type):
                    continue
                
                # Convert to string and clean
                meter_id_str = str(meter_id).strip()
                incident_type_str = str(incident_type).strip()
                
                # Skip header row (if meter_id looks like a header)
                if (meter_id_str.upper() in ['ID MEDIDOR', 'MEDIDOR', 'METER ID', 'METER_ID'] or 
                    incident_type_str.upper() in ['INCIDENCIA/SIN LECTURA', 'INCIDENCIA', 'SIN LECTURA']):
                    continue
                
                if meter_id_str and incident_type_str:
                    incidents_dict[meter_id_str] = incident_type_str
                    
            except (IndexError, ValueError, KeyError) as e:
                # Skip rows with missing data
                continue
        
        print(f"✅ Archivo de incidencias procesado: {len(incidents_dict)} medidores con incidencias")
        
        return incidents_dict
        
    except Exception as e:
        print(f"❌ Error leyendo archivo de incidencias: {str(e)}")
        return {}

def add_missing_meters_to_pivot(pivot_df, all_meters_set, incidents_dict=None, customer_mapping=None):
    """
    Add missing meters to the pivot table with "SIN LECTURA" classification
    and incident information
    """
    if not all_meters_set or pivot_df.empty:
        return pivot_df
    
    # Find meter column in pivot table
    meter_col = None
    for col in pivot_df.columns:
        if 'meter id' in col.lower() or 'meter_id' in col.lower():
            meter_col = col
            break
    
    if meter_col is None:
        print("⚠️ No se encontró columna de medidor en tabla dinámica")
        return pivot_df
    
    # Get existing meters in pivot table
    existing_meters = set(pivot_df[meter_col].astype(str).unique())
    
    # Find missing meters
    missing_meters = all_meters_set - existing_meters
    
    if not missing_meters:
        return pivot_df
    
    # Get date columns (excluding meter, customer, summary, classification, and incident columns)
    exclude_cols = [meter_col, 'Customer ID', 'Numero De Lecturas', 'Dias Lecturadas', 'Clasificacion', 'Incidencia', 'Detalle de Incidencia']
    date_columns = [col for col in pivot_df.columns if col not in exclude_cols]
    
    # Create rows for missing meters
    missing_rows = []
    for meter_id in missing_meters:
        row = {meter_col: meter_id}
        
        # Add customer ID from customer mapping or default to 'SIN CLIENTE'
        if customer_mapping and meter_id in customer_mapping:
            row['Customer ID'] = customer_mapping[meter_id]
        else:
            row['Customer ID'] = 'SIN CLIENTE'
        
        # Set all date columns to 0 (no readings)
        for date_col in date_columns:
            row[date_col] = 0
        # Set summary columns
        row['Numero De Lecturas'] = 0
        row['Dias Lecturadas'] = 0
        row['Clasificacion'] = 'SIN LECTURA'
        
        # Add incident information
        if incidents_dict and meter_id in incidents_dict:
            row['Incidencia'] = 'CON INCIDENCIA'
            row['Detalle de Incidencia'] = incidents_dict[meter_id]
        else:
            row['Incidencia'] = 'SIN INCIDENCIA'
            row['Detalle de Incidencia'] = 'SIN INCIDENCIA'
            
        missing_rows.append(row)
    
    # Create DataFrame for missing meters
    missing_df = pd.DataFrame(missing_rows)
    
    # Ensure column order matches original pivot table
    missing_df = missing_df[pivot_df.columns]
    
    # Combine with original pivot table
    combined_df = pd.concat([pivot_df, missing_df], ignore_index=True)
    
    # Sort to put "SIN LECTURA" meters at the very bottom
    combined_df['sort_order'] = combined_df['Clasificacion'].apply(
        lambda x: 0 if x in ['DIARIO', 'INTERMITENTE'] 
                else 1 if x in ['DIARIO / WALKBY', 'INTERMITENTE / WALKBY']
                else 2 if x == 'WALKBY'
                else 3  # SIN LECTURA goes last
    )
    combined_df = combined_df.sort_values(['sort_order', meter_col]).reset_index(drop=True)
    combined_df = combined_df.drop('sort_order', axis=1)
    
    return combined_df

def create_pivot_table(df_with_gateway, df_without_gateway=None, incidents_dict=None, customer_mapping=None):
    """
    Create a pivot table with Meter ID as rows, Record Time as columns, and total flow count as values
    Plus sum and count columns at the end, classification based on reading frequency, and incident information
    """
    # Find the required columns
    meter_col = None
    customer_col = None
    date_col = None
    flow_col = None
    
    for col in df_with_gateway.columns:
        col_lower = col.lower()
        if 'meter id' in col_lower or 'meter_id' in col_lower:
            meter_col = col
        elif 'customer' in col_lower or 'client' in col_lower or 'cliente' in col_lower:
            customer_col = col
        elif 'record time' in col_lower and 'time_time' not in col_lower:
            date_col = col
        elif 'flow' in col_lower and ('total' in col_lower or 'count' in col_lower or col_lower == 'flow'):
            flow_col = col
    
    # If no specific flow column found, look for any numeric column that might be flow data
    if flow_col is None:
        numeric_cols = df_with_gateway.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'flow' in col.lower() or 'count' in col.lower() or 'volume' in col.lower():
                flow_col = col
                break
        
        # If still not found, use the first numeric column
        if flow_col is None and len(numeric_cols) > 0:
            flow_col = numeric_cols[0]
    
    if not all([meter_col, date_col, flow_col]):
        print("❌ Error: No se pudieron encontrar las columnas requeridas para la tabla dinámica")
        return pd.DataFrame()
    
    try:
        # Get list of walkby meter IDs for classification
        walkby_meter_ids = set()
        if df_without_gateway is not None and not df_without_gateway.empty and meter_col in df_without_gateway.columns:
            walkby_meter_ids = set(df_without_gateway[meter_col].unique())
        
        # Create pivot table with count instead of sum
        pivot_df = df_with_gateway.pivot_table(
            index=meter_col,
            columns=date_col,
            values=flow_col,
            aggfunc='count',  # Changed from 'sum' to 'count'
            fill_value=0
        )
        
        # Convert column names to strings to avoid datetime.date issues
        pivot_df.columns = pivot_df.columns.astype(str)
        
        # Also create pivot table for walkby data if it exists
        walkby_pivot_df = None
        if df_without_gateway is not None and not df_without_gateway.empty and meter_col in df_without_gateway.columns:
            # Check if walkby data has the same columns
            if date_col in df_without_gateway.columns and flow_col in df_without_gateway.columns:
                walkby_pivot_df = df_without_gateway.pivot_table(
                    index=meter_col,
                    columns=date_col,
                    values=flow_col,
                    aggfunc='count',
                    fill_value=0
                )
                
                # Convert column names to strings to avoid datetime.date issues
                walkby_pivot_df.columns = walkby_pivot_df.columns.astype(str)
        
        # Combine gateway and walkby pivot tables
        if walkby_pivot_df is not None and not walkby_pivot_df.empty:
            # Align columns (dates) between both pivot tables
            all_dates = set(pivot_df.columns) | set(walkby_pivot_df.columns)
            
            # Add missing columns to both dataframes
            for date in all_dates:
                date_str = str(date)  # Ensure string representation
                if date_str not in pivot_df.columns:
                    pivot_df[date_str] = 0
                if date_str not in walkby_pivot_df.columns:
                    walkby_pivot_df[date_str] = 0
            
            # Reorder columns to match
            date_cols_sorted = sorted(all_dates, key=lambda x: pd.to_datetime(str(x), errors='coerce'))
            # Convert to strings to ensure consistency
            date_cols_sorted = [str(col) for col in date_cols_sorted]
            pivot_df = pivot_df[date_cols_sorted]
            walkby_pivot_df = walkby_pivot_df[date_cols_sorted]
            
            # Store walkby data separately for later filtering
            walkby_data_backup = walkby_pivot_df.copy()
            
            # Combine the pivot tables (walkby data will add to existing or create new rows)
            combined_pivot = pivot_df.add(walkby_pivot_df, fill_value=0)
            pivot_df = combined_pivot
        
        # Reset index to make Meter ID a regular column
        pivot_df = pivot_df.reset_index()
        
        # Add customer information
        final_customer_mapping = {}
        if customer_mapping:
            final_customer_mapping.update(customer_mapping)
        
        # Add customer info from readings data if customer column exists and not already mapped
        if customer_col is not None:
            # Create customer mapping from readings data
            readings_customer_mapping = {}
            
            # Get customer info from gateway data
            if not df_with_gateway.empty and meter_col in df_with_gateway.columns:
                gateway_customer_map = df_with_gateway.groupby(meter_col)[customer_col].first().to_dict()
                readings_customer_mapping.update(gateway_customer_map)
            
            # Get customer info from walkby data if available
            if df_without_gateway is not None and not df_without_gateway.empty and meter_col in df_without_gateway.columns and customer_col in df_without_gateway.columns:
                walkby_customer_map = df_without_gateway.groupby(meter_col)[customer_col].first().to_dict()
                readings_customer_mapping.update(walkby_customer_map)
            
            # Add readings customer info only for meters not already mapped from meters file
            for meter_id, customer_id in readings_customer_mapping.items():
                if meter_id not in final_customer_mapping and pd.notna(customer_id):
                    final_customer_mapping[meter_id] = customer_id
        
        # Add customer column to pivot table
        pivot_df['Customer ID'] = pivot_df[meter_col].map(final_customer_mapping).fillna('SIN CLIENTE')
        
        # Sort columns (dates) chronologically
        date_columns = [col for col in pivot_df.columns if col not in [meter_col, 'Customer ID']]
        if date_columns:
            try:
                # Try to sort dates chronologically
                date_columns_sorted = sorted(date_columns, key=lambda x: pd.to_datetime(str(x), errors='coerce'))
                # Keep only actual date columns for the pivot data
                actual_date_columns = [col for col in date_columns_sorted if col not in ['Numero De Lecturas', 'Dias Lecturadas', 'Clasificacion', 'Incidencia', 'Detalle de Incidencia']]
                pivot_df = pivot_df[[meter_col, 'Customer ID'] + actual_date_columns]
                date_columns = actual_date_columns
            except:
                # If sorting fails, keep original order
                actual_date_columns = [col for col in date_columns if col not in ['Numero De Lecturas', 'Dias Lecturadas', 'Clasificacion', 'Incidencia', 'Detalle de Incidencia']]
                date_columns = actual_date_columns
        
        # Add sum of counts column (total readings per meter)
        pivot_df['Numero De Lecturas'] = pivot_df[date_columns].sum(axis=1)
        
        # Add count of counts column (number of days with readings per meter)
        pivot_df['Dias Lecturadas'] = (pivot_df[date_columns] > 0).sum(axis=1)
        
        # Calculate month-specific classification
        if date_columns:
            # Group dates by month to handle multi-month data correctly
            month_groups = {}
            for date_str in date_columns:
                try:
                    date_obj = pd.to_datetime(str(date_str), errors='coerce')
                    if not pd.isna(date_obj):
                        month_key = (date_obj.year, date_obj.month)
                        if month_key not in month_groups:
                            month_groups[month_key] = []
                        month_groups[month_key].append(str(date_str))
                except:
                    continue
        else:
            month_groups = {}
        
        # Add classification column with month-aware logic
        def classify_meter(row):
            count_of_counts = row['Dias Lecturadas']
            meter_id = row[meter_col]
            is_walkby = meter_id in walkby_meter_ids
            
            # Check if this meter exists in gateway data
            gateway_meter_ids = set()
            if not df_with_gateway.empty and meter_col in df_with_gateway.columns:
                gateway_meter_ids = set(df_with_gateway[meter_col].unique())
            
            is_pure_walkby = is_walkby and meter_id not in gateway_meter_ids
            
            # Pure walkby meters get special classification
            if is_pure_walkby:
                return "WALKBY"
            
            # For mixed meters (gateway + walkby) or pure gateway meters
            if month_groups:
                is_daily_overall = True
                for (year, month), month_dates in month_groups.items():
                    # Count readings for this meter in this specific month
                    month_readings = sum(1 for date_str in month_dates if row[date_str] > 0)
                    
                    # Calculate total days in this month
                    if month == 12:
                        next_month = datetime(year + 1, 1, 1)
                    else:
                        next_month = datetime(year, month + 1, 1)
                    last_day = next_month - timedelta(days=1)
                    days_in_month = last_day.day
                    
                    # If meter doesn't have readings for all days in ANY month, it's not daily
                    if month_readings < days_in_month:
                        is_daily_overall = False
                        break
                
                if is_daily_overall:
                    return "DIARIO / WALKBY" if is_walkby else "DIARIO"
                else:
                    return "INTERMITENTE / WALKBY" if is_walkby else "INTERMITENTE"
            else:
                # Fallback to simple comparison if no month data available
                # Assume 30 days for classification
                if count_of_counts >= 30:
                    return "DIARIO / WALKBY" if is_walkby else "DIARIO"
                else:
                    return "INTERMITENTE / WALKBY" if is_walkby else "INTERMITENTE"
        
        pivot_df['Clasificacion'] = pivot_df.apply(classify_meter, axis=1)
        
        # Add incident information columns
        if incidents_dict:
            def add_incident_info(row):
                meter_id = str(row[meter_col])
                if meter_id in incidents_dict:
                    return 'CON INCIDENCIA', incidents_dict[meter_id]
                else:
                    return 'SIN INCIDENCIA', 'SIN INCIDENCIA'
            
            # Apply incident information
            incident_info = pivot_df.apply(add_incident_info, axis=1, result_type='expand')
            pivot_df['Incidencia'] = incident_info[0]
            pivot_df['Detalle de Incidencia'] = incident_info[1]
            
            # Remove walkby readings for meters with incidents
            if 'walkby_data_backup' in locals() and not walkby_data_backup.empty:
                meters_with_incidents = set()
                for _, row in pivot_df.iterrows():
                    if row['Incidencia'] == 'CON INCIDENCIA':
                        meter_id = row[meter_col]
                        # Check if this meter has both gateway and walkby data
                        if (meter_id in walkby_meter_ids and 
                            not df_with_gateway.empty and 
                            meter_id in df_with_gateway[meter_col].values):
                            meters_with_incidents.add(meter_id)
                
                if meters_with_incidents:
                    # Remove walkby data for these meters
                    for meter_id in meters_with_incidents:
                        # Find the row for this meter
                        meter_mask = pivot_df[meter_col] == meter_id
                        meter_row_idx = pivot_df[meter_mask].index
                        
                        if len(meter_row_idx) > 0:
                            row_idx = meter_row_idx[0]
                            
                            # Subtract walkby data from combined data for this meter
                            if meter_id in walkby_data_backup.index:
                                walkby_row = walkby_data_backup.loc[meter_id]
                                
                                # Subtract walkby readings from each date column
                                for date_col in date_columns:
                                    if date_col in walkby_row.index and not pd.isna(walkby_row[date_col]):
                                        current_value = pivot_df.loc[row_idx, date_col]
                                        walkby_value = int(walkby_row[date_col])
                                        new_value = max(0, current_value - walkby_value)
                                        pivot_df.loc[row_idx, date_col] = new_value
                    
                    # Recalculate sum and count columns after removing walkby data
                    affected_rows = pivot_df[pivot_df[meter_col].isin(meters_with_incidents)]
                    for idx in affected_rows.index:
                        pivot_df.loc[idx, 'Numero De Lecturas'] = pivot_df.loc[idx, date_columns].sum()
                        pivot_df.loc[idx, 'Dias Lecturadas'] = (pivot_df.loc[idx, date_columns] > 0).sum()
                    
                    # Update walkby_meter_ids to exclude meters with incidents
                    walkby_meter_ids_updated = walkby_meter_ids - meters_with_incidents
                    
                    def reclassify_meter(row):
                        count_of_counts = row['Dias Lecturadas']
                        meter_id = row[meter_col]
                        has_incident = row['Incidencia'] == 'CON INCIDENCIA'
                        
                        # Use updated walkby list (excluding meters with incidents)
                        is_walkby = meter_id in walkby_meter_ids_updated
                        
                        # Check if this meter exists in gateway data
                        gateway_meter_ids = set()
                        if not df_with_gateway.empty and meter_col in df_with_gateway.columns:
                            gateway_meter_ids = set(df_with_gateway[meter_col].unique())
                        
                        is_pure_walkby = is_walkby and meter_id not in gateway_meter_ids
                        
                        # Pure walkby meters get special classification
                        if is_pure_walkby:
                            return "WALKBY"
                        
                        # For gateway meters (including those that had walkby removed due to incidents)
                        if month_groups:
                            is_daily_overall = True
                            for (year, month), month_dates in month_groups.items():
                                # Count readings for this meter in this specific month
                                month_readings = sum(1 for date_str in month_dates if row[date_str] > 0)
                                
                                # Calculate total days in this month
                                if month == 12:
                                    next_month = datetime(year + 1, 1, 1)
                                else:
                                    next_month = datetime(year, month + 1, 1)
                                last_day = next_month - timedelta(days=1)
                                days_in_month = last_day.day
                                
                                # If meter doesn't have readings for all days in ANY month, it's not daily
                                if month_readings < days_in_month:
                                    is_daily_overall = False
                                    break
                            
                            if is_daily_overall:
                                return "DIARIO / WALKBY" if is_walkby else "DIARIO"
                            else:
                                return "INTERMITENTE / WALKBY" if is_walkby else "INTERMITENTE"
                        else:
                            # Fallback to simple comparison if no month data available
                            # Assume 30 days for classification
                            if count_of_counts >= 30:
                                return "DIARIO / WALKBY" if is_walkby else "DIARIO"
                            else:
                                return "INTERMITENTE / WALKBY" if is_walkby else "INTERMITENTE"
                    
                    # Recalculate classification for all meters using updated logic
                    pivot_df['Clasificacion'] = pivot_df.apply(reclassify_meter, axis=1)
            
            # Additional processing: Remove walkby readings for DIARIO/WALKBY meters when they conflict with gateway readings
            if 'walkby_data_backup' in locals() and not walkby_data_backup.empty:
                # Find DIARIO/WALKBY meters
                diario_walkby_meters = set()
                for _, row in pivot_df.iterrows():
                    if row['Clasificacion'] == 'DIARIO / WALKBY':
                        meter_id = row[meter_col]
                        diario_walkby_meters.add(meter_id)
                
                if diario_walkby_meters:
                    # Check for conflicts and remove walkby readings on days with gateway readings
                    meters_with_conflicts = set()
                    
                    for meter_id in diario_walkby_meters:
                        # Find the row for this meter in pivot table
                        meter_mask = pivot_df[meter_col] == meter_id
                        meter_row_idx = pivot_df[meter_mask].index
                        
                        if len(meter_row_idx) > 0:
                            row_idx = meter_row_idx[0]
                            
                            # Check if this meter has walkby data
                            if meter_id in walkby_data_backup.index:
                                walkby_row = walkby_data_backup.loc[meter_id]
                                
                                # Check each date column for conflicts
                                has_conflicts = False
                                for date_col in date_columns:
                                    if date_col in walkby_row.index and not pd.isna(walkby_row[date_col]):
                                        walkby_count = int(walkby_row[date_col])
                                        total_count = pivot_df.loc[row_idx, date_col]
                                        gateway_count = total_count - walkby_count
                                        
                                        # If there are both gateway and walkby readings on the same day
                                        if gateway_count > 0 and walkby_count > 0:
                                            has_conflicts = True
                                            # Remove walkby readings for this day
                                            pivot_df.loc[row_idx, date_col] = gateway_count
                                
                                if has_conflicts:
                                    meters_with_conflicts.add(meter_id)
                    
                    # Recalculate totals for affected meters
                    if meters_with_conflicts:
                        affected_mask = pivot_df[meter_col].isin(meters_with_conflicts)
                        for idx in pivot_df[affected_mask].index:
                            pivot_df.loc[idx, 'Numero De Lecturas'] = pivot_df.loc[idx, date_columns].sum()
                            pivot_df.loc[idx, 'Dias Lecturadas'] = (pivot_df.loc[idx, date_columns] > 0).sum()

                    # --- FINAL RE-CLASSIFICATION STEP ---
                    # This now runs AFTER all conflicts have been resolved and totals recalculated.
                    print("🔄 Reclasificando medidores DIARIO/WALKBY según política de la empresa...")
                    
                    # Step 1: Reclassify meters with conflicts to DIARIO
                    if meters_with_conflicts:
                        conflict_mask = (pivot_df['Clasificacion'] == 'DIARIO / WALKBY') & (pivot_df[meter_col].isin(meters_with_conflicts))
                        for meter_id in pivot_df.loc[conflict_mask, meter_col]:
                            print(f"   🔄 Medidor {meter_id}: DIARIO/WALKBY → DIARIO (conflictos resueltos)")
                        pivot_df.loc[conflict_mask, 'Clasificacion'] = 'DIARIO'

                    # Step 2: Reclassify all remaining DIARIO/WALKBY meters to INTERMITENTE/WALKBY
                    # These are the ones that had no conflicts.
                    intermitente_mask = pivot_df['Clasificacion'] == 'DIARIO / WALKBY'
                    if intermitente_mask.any():
                        for meter_id in pivot_df.loc[intermitente_mask, meter_col]:
                            print(f"   🔄 Medidor {meter_id}: DIARIO/WALKBY → INTERMITENTE/WALKBY (política empresa)")
                        pivot_df.loc[intermitente_mask, 'Clasificacion'] = 'INTERMITENTE / WALKBY'
            
        else:
            # Add empty incident columns if no incidents file provided
            pivot_df['Incidencia'] = 'SIN INCIDENCIA'
            pivot_df['Detalle de Incidencia'] = 'SIN INCIDENCIA'
        
        # Sort to put WALKBY meters at the bottom
        pivot_df['sort_order'] = pivot_df['Clasificacion'].apply(
            lambda x: 0 if x in ['DIARIO', 'INTERMITENTE'] 
                    else 1 if x in ['DIARIO / WALKBY', 'INTERMITENTE / WALKBY']
                    else 2  # Pure WALKBY goes last
        )
        pivot_df = pivot_df.sort_values(['sort_order', meter_col]).reset_index(drop=True)
        pivot_df = pivot_df.drop('sort_order', axis=1)
        
        # Final column reordering
        # Ensure all date columns are strings
        date_columns = [str(col) for col in date_columns]
        
        # Define the desired column order
        ordered_columns = []
        ordered_columns.append(meter_col)
        ordered_columns.append('Customer ID')
        ordered_columns.append('Clasificacion')
        ordered_columns.append('Incidencia')
        ordered_columns.append('Detalle de Incidencia')
        ordered_columns.extend(date_columns)
        ordered_columns.append('Numero De Lecturas')
        ordered_columns.append('Dias Lecturadas')
        
        # Reorder the dataframe columns
        pivot_df = pivot_df[ordered_columns]
        
        return pivot_df
        
    except Exception as e:
        print(f"❌ Error creando tabla dinámica: {str(e)}")
        return pd.DataFrame()

def read_files_parallel(data_file_path, meters_file_path=None, incidents_file_path=None):
    """
    Read multiple files in parallel for improved performance
    """
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks for parallel execution
        futures = []
        
        # Main data file reading
        if data_file_path.endswith('.xls'):
            data_future = executor.submit(pd.read_excel, data_file_path, sheet_name=None, engine='xlrd')
        else:
            data_future = executor.submit(pd.read_excel, data_file_path, sheet_name=None, engine='openpyxl')
        futures.append(('data', data_future))
        
        # Meters file reading
        meters_future = None
        if meters_file_path:
            meters_future = executor.submit(read_meters_file, meters_file_path)
            futures.append(('meters', meters_future))
        
        # Incidents file reading  
        incidents_future = None
        if incidents_file_path:
            incidents_future = executor.submit(read_incidents_file, incidents_file_path)
            futures.append(('incidents', incidents_future))
        
        # Collect results
        results = {}
        for name, future in futures:
            try:
                results[name] = future.result()
                print(f"✅ Archivo {name} leído exitosamente")
            except Exception as e:
                print(f"❌ Error leyendo archivo {name}: {str(e)}")
                results[name] = None
        
        # Extract results
        all_sheets = results.get('data', {})
        
        if meters_future:
            meters_result = results.get('meters', (set(), {}))
            all_meters_set = meters_result[0] if meters_result else set()
            customer_mapping = meters_result[1] if meters_result else {}
        else:
            all_meters_set = set()
            customer_mapping = {}
            
        if incidents_future:
            incidents_dict = results.get('incidents', {})
        else:
            incidents_dict = {}
    
    return all_sheets, all_meters_set, customer_mapping, incidents_dict

def process_dataframes_parallel(dataframes):
    """
    Process multiple dataframes in parallel for improved performance
    """
    
    if len(dataframes) <= 1:
        return dataframes
    
    # Split dataframes into chunks for parallel processing
    chunk_size = max(1, len(dataframes) // min(4, len(dataframes)))
    chunks = [dataframes[i:i + chunk_size] for i in range(0, len(dataframes), chunk_size)]
    
    def process_chunk(chunk):
        """Process a chunk of dataframes"""
        processed = []
        for df in chunk:
            if not df.empty:
                # Apply basic optimizations
                df = split_datetime_column(df)
                processed.append(df)
        return processed
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Process chunks in parallel
        future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
        
        processed_dataframes = []
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                result = future.result()
                processed_dataframes.extend(result)
            except Exception as e:
                print(f"⚠️ Error procesando chunk: {str(e)}")
    
    return processed_dataframes

def append_all_sheets_optimized(data_file_path, meters_file_path=None, incidents_file_path=None):
    """
    Optimized file processing with chunking, parallel processing and efficient operations
    Now supports optional meters file for complete system coverage and incidents file
    """
    print("🔄 Procesando archivo...")
    start_time = datetime.now()
    
    try:
        # Read all files in parallel for better performance
        all_sheets, all_meters_set, customer_mapping, incidents_dict = read_files_parallel(
            data_file_path, meters_file_path, incidents_file_path
        )
        
        # Create output filename with dep_ prefix and ensure .xlsx extension
        input_dir = os.path.dirname(data_file_path)
        input_filename = os.path.splitext(os.path.basename(data_file_path))[0]
        # Always use .xlsx extension for output since we're using modern Excel engines
        xlsx_path = os.path.join(input_dir, f"dep_{input_filename}.xlsx")
        
        # Efficient concatenation using parallel processing
        dataframes = [df for df in all_sheets.values() if not df.empty]
        
        if not dataframes:
            print("❌ No se encontraron datos válidos")
            return
        
        # Process dataframes in parallel before concatenation
        processed_dataframes = process_dataframes_parallel(dataframes)
        
        # Single concatenation operation
        combined_df = pd.concat(processed_dataframes, ignore_index=True, sort=False)
        
        # Process in parallel pipeline
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both operations in parallel
            split_future = executor.submit(split_by_gateway_optimized, combined_df)
            
            # Wait for split operation to complete
            df_with_gateway, df_without_gateway = split_future.result()
        
        # Use optimized limiting function with parallel processing
        df_with_gateway_limited, deleted_df = limit_readings_per_day_optimized(df_with_gateway, max_readings=7)
        
        # Filter WALKBY data to keep only the latest reading per meter per day
        df_without_gateway_filtered, deleted_walkby_duplicates = filter_walkby_latest_per_day(df_without_gateway)
        df_without_gateway = df_without_gateway_filtered
        
        # Filter pure WALKBY meters to keep only first reading per month
        df_without_gateway_monthly_filtered, deleted_walkby_monthly = limit_pure_walkby_to_first_per_month(df_without_gateway, df_with_gateway_limited)
        df_without_gateway = df_without_gateway_monthly_filtered
        
        # Initialize deleted WALKBY readings tracking
        deleted_walkby_df = pd.DataFrame()
        
        # Add deleted WALKBY duplicates to tracking
        if not deleted_walkby_duplicates.empty:
            deleted_walkby_df = pd.concat([deleted_walkby_df, deleted_walkby_duplicates], ignore_index=True)
        
        # Add deleted monthly WALKBY readings to tracking
        if not deleted_walkby_monthly.empty:
            deleted_walkby_df = pd.concat([deleted_walkby_df, deleted_walkby_monthly], ignore_index=True)
        
        # Filter walkby data based on incidents BEFORE creating pivot table
        if incidents_dict and not df_without_gateway.empty:
            # Find meter column in walkby data
            meter_col_walkby = None
            for col in df_without_gateway.columns:
                col_lower = col.lower()
                if 'meter id' in col_lower or 'meter_id' in col_lower:
                    meter_col_walkby = col
                    break
            
            if meter_col_walkby:
                # Get meters with incidents that have walkby data
                meters_to_remove_from_walkby = set()
                for meter_id in incidents_dict.keys():
                    # Check if this meter has walkby data
                    if meter_id in df_without_gateway[meter_col_walkby].astype(str).values:
                        # Check if it also has gateway data
                        has_gateway_data = (not df_with_gateway_limited.empty and 
                                          meter_id in df_with_gateway_limited[meter_col_walkby].astype(str).values)
                        meters_to_remove_from_walkby.add(meter_id)
                
                if meters_to_remove_from_walkby:
                    
                    # Collect deleted WALKBY readings due to incidents
                    deleted_walkby_incidents = df_without_gateway[
                        df_without_gateway[meter_col_walkby].astype(str).isin(meters_to_remove_from_walkby)
                    ].copy()
                    
                    # Add deletion reason
                    deleted_walkby_incidents['Motivo_Eliminacion'] = 'Medidor con incidencia'
                    
                    # Filter out these meters from walkby data
                    df_without_gateway = df_without_gateway[
                        ~df_without_gateway[meter_col_walkby].astype(str).isin(meters_to_remove_from_walkby)
                    ].copy()
                    
                    # Add to deleted WALKBY tracking
                    deleted_walkby_df = pd.concat([deleted_walkby_df, deleted_walkby_incidents], ignore_index=True)
        
        # Create pivot table from the limited gateway data and filtered walkby data
        
        # Use parallel processing for pivot table creation and VALORIZABLE preparation
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit pivot table creation
            pivot_future = executor.submit(create_pivot_table, df_with_gateway_limited, df_without_gateway, incidents_dict, customer_mapping)
            
            # Wait for pivot table to complete
            pivot_table_df = pivot_future.result()
        
        # Additional filtering: Remove WALKBY readings for DIARIO/WALKBY meters on days with Gateway conflicts
        if not pivot_table_df.empty and not df_without_gateway.empty:
            # Find meter and date columns in walkby data
            meter_col_walkby = None
            date_col_walkby = None
            for col in df_without_gateway.columns:
                col_lower = col.lower()
                if 'meter id' in col_lower or 'meter_id' in col_lower:
                    meter_col_walkby = col
                elif 'record time' in col_lower and 'time_time' not in col_lower:
                    date_col_walkby = col
            
            if meter_col_walkby and date_col_walkby:
                # Get meters that were reclassified from DIARIO/WALKBY to DIARIO due to conflicts
                reclassified_meters = set()
                diario_meters = pivot_table_df[pivot_table_df['Clasificacion'] == 'DIARIO']
                
                for _, row in diario_meters.iterrows():
                    meter_id = str(row[meter_col_walkby])  # Use same column name
                    # Check if this meter originally had WALKBY data
                    if meter_id in df_without_gateway[meter_col_walkby].astype(str).values:
                        reclassified_meters.add(meter_id)
                
                if reclassified_meters:
                    
                    # Collect deleted WALKBY readings due to conflicts
                    deleted_walkby_conflicts = df_without_gateway[
                        df_without_gateway[meter_col_walkby].astype(str).isin(reclassified_meters)
                    ].copy()
                    
                    # Add deletion reason
                    deleted_walkby_conflicts['Motivo_Eliminacion'] = 'Conflicto con lectura Gateway'
                    
                    # Remove WALKBY readings for these meters
                    df_without_gateway = df_without_gateway[
                        ~df_without_gateway[meter_col_walkby].astype(str).isin(reclassified_meters)
                    ].copy()
                    
                    # Add to deleted WALKBY tracking
                    deleted_walkby_df = pd.concat([deleted_walkby_df, deleted_walkby_conflicts], ignore_index=True)
        
        # Add missing meters if meters file was provided
        if all_meters_set and not pivot_table_df.empty:
            pivot_table_df = add_missing_meters_to_pivot(pivot_table_df, all_meters_set, incidents_dict, customer_mapping)
        
        # Handle pure WALKBY meters with incidents that were removed but need to appear in pivot table
        if incidents_dict and not pivot_table_df.empty:
            # Find meter column in pivot table
            meter_col = None
            for col in pivot_table_df.columns:
                if 'meter id' in col.lower() or 'meter_id' in col.lower():
                    meter_col = col
                    break
            
            if meter_col:
                # Get existing meters in pivot table
                existing_meters = set(pivot_table_df[meter_col].astype(str).unique())
                
                # Find pure WALKBY meters with incidents that were removed
                pure_walkby_with_incidents = []
                for meter_id in incidents_dict.keys():
                    if meter_id not in existing_meters:
                        # This meter with incident is not in pivot table, it was likely a pure WALKBY meter
                        pure_walkby_with_incidents.append(meter_id)
                
                if pure_walkby_with_incidents:
                    # Get date columns (excluding meter, customer, summary, classification, and incident columns)
                    exclude_cols = [meter_col, 'Customer ID', 'Numero De Lecturas', 'Dias Lecturadas', 'Clasificacion', 'Incidencia', 'Detalle de Incidencia']
                    date_columns = [col for col in pivot_table_df.columns if col not in exclude_cols]
                    
                    # Create rows for pure WALKBY meters with incidents
                    walkby_incident_rows = []
                    for meter_id in pure_walkby_with_incidents:
                        row = {meter_col: meter_id}
                        
                        # Add customer ID from customer mapping or default to 'SIN CLIENTE'
                        if customer_mapping and meter_id in customer_mapping:
                            row['Customer ID'] = customer_mapping[meter_id]
                        else:
                            row['Customer ID'] = 'SIN CLIENTE'
                        
                        # Set all date columns to 0 (no readings)
                        for date_col in date_columns:
                            row[date_col] = 0
                        # Set summary columns
                        row['Numero De Lecturas'] = 0
                        row['Dias Lecturadas'] = 0
                        row['Clasificacion'] = 'SIN LECTURA'
                        
                        # Add incident information
                        row['Incidencia'] = 'CON INCIDENCIA'
                        row['Detalle de Incidencia'] = incidents_dict[meter_id]
                        
                        walkby_incident_rows.append(row)
                    
                    # Create DataFrame for pure WALKBY meters with incidents
                    walkby_incident_df = pd.DataFrame(walkby_incident_rows)
                    
                    # Ensure column order matches original pivot table
                    walkby_incident_df = walkby_incident_df[pivot_table_df.columns]
                    
                    # Combine with original pivot table
                    pivot_table_df = pd.concat([pivot_table_df, walkby_incident_df], ignore_index=True)
                    
                    # Sort to put "SIN LECTURA" meters at the very bottom
                    pivot_table_df['sort_order'] = pivot_table_df['Clasificacion'].apply(
                        lambda x: 0 if x in ['DIARIO', 'INTERMITENTE'] 
                                else 1 if x in ['DIARIO / WALKBY', 'INTERMITENTE / WALKBY']
                                else 2 if x == 'WALKBY'
                                else 3  # SIN LECTURA goes last
                    )
                    pivot_table_df = pivot_table_df.sort_values(['sort_order', meter_col]).reset_index(drop=True)
                    pivot_table_df = pivot_table_df.drop('sort_order', axis=1)
        
        # Combine deleted Gateway and WALKBY readings for the "Eliminados" sheet
        all_deleted_df = pd.DataFrame()
        
        # Add Gateway deletions (with reason)
        if not deleted_df.empty:
            gateway_deleted = deleted_df.copy()
            gateway_deleted['Motivo_Eliminacion'] = 'Más de 7 lecturas por día'
            all_deleted_df = pd.concat([all_deleted_df, gateway_deleted], ignore_index=True)
        
        # Add WALKBY deletions (already have reason)
        if not deleted_walkby_df.empty:
            all_deleted_df = pd.concat([all_deleted_df, deleted_walkby_df], ignore_index=True)
        
        # Update deleted_df to include all deletions
        deleted_df = all_deleted_df
        
        # Create VALORIZABLE sheet from final processed data AFTER all processing is complete
        
        # Use parallel processing for VALORIZABLE data preparation - now split into GATEWAY and WALKBY
        def prepare_valorizable_data():
            valorizable_gateway_df = pd.DataFrame()
            valorizable_walkby_df = pd.DataFrame()
            
            if not pivot_table_df.empty:
                # Find meter column in pivot table
                meter_col_pivot = None
                for col in pivot_table_df.columns:
                    if 'meter id' in col.lower() or 'meter_id' in col.lower():
                        meter_col_pivot = col
                        break
                
                # Find meter column in processed data
                meter_col_gateway = None
                meter_col_walkby = None
                
                if not df_with_gateway_limited.empty:
                    for col in df_with_gateway_limited.columns:
                        if 'meter id' in col.lower() or 'meter_id' in col.lower():
                            meter_col_gateway = col
                            break
                
                if not df_without_gateway.empty:
                    for col in df_without_gateway.columns:
                        if 'meter id' in col.lower() or 'meter_id' in col.lower():
                            meter_col_walkby = col
                            break
                
                if meter_col_pivot:
                    # Get meters that meet valorizable conditions from pivot table
                    
                    # Condition 1: DIARIO with SIN INCIDENCIA
                    diario_meters = pivot_table_df[
                        (pivot_table_df['Clasificacion'] == 'DIARIO') & 
                        (pivot_table_df['Incidencia'] == 'SIN INCIDENCIA')
                    ][meter_col_pivot].astype(str).tolist()
                    
                    # Condition 2: WALKBY with SIN INCIDENCIA  
                    walkby_meters = pivot_table_df[
                        (pivot_table_df['Clasificacion'] == 'WALKBY') & 
                        (pivot_table_df['Incidencia'] == 'SIN INCIDENCIA')
                    ][meter_col_pivot].astype(str).tolist()
                    
                    # Condition 3: INTERMITENTE / WALKBY with SIN INCIDENCIA (only WALKBY readings)
                    intermitente_walkby_meters = pivot_table_df[
                        (pivot_table_df['Clasificacion'] == 'INTERMITENTE / WALKBY') & 
                        (pivot_table_df['Incidencia'] == 'SIN INCIDENCIA')
                    ][meter_col_pivot].astype(str).tolist()
                    
                    # Collect valorizable readings separated by type using parallel processing
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as sub_executor:
                        # Prepare tasks for parallel execution
                        tasks = []
                        
                        # Task 1: Gateway readings for DIARIO meters only
                        if diario_meters and meter_col_gateway:
                            task1 = sub_executor.submit(
                                lambda: df_with_gateway_limited[
                                    df_with_gateway_limited[meter_col_gateway].astype(str).isin(diario_meters)
                                ].copy()
                            )
                            tasks.append(('gateway_diario', task1))
                        
                        # Task 2: Walkby readings for DIARIO meters 
                        if diario_meters and meter_col_walkby:
                            task2 = sub_executor.submit(
                                lambda: df_without_gateway[
                                    df_without_gateway[meter_col_walkby].astype(str).isin(diario_meters)
                                ].copy()
                            )
                            tasks.append(('walkby_diario', task2))
                        
                        # Task 3: Pure WALKBY meters
                        if walkby_meters and meter_col_walkby:
                            task3 = sub_executor.submit(
                                lambda: df_without_gateway[
                                    df_without_gateway[meter_col_walkby].astype(str).isin(walkby_meters)
                                ].copy()
                            )
                            tasks.append(('walkby_pure', task3))
                        
                        # Task 4: INTERMITENTE/WALKBY meters (only walkby readings)
                        if intermitente_walkby_meters and meter_col_walkby:
                            task4 = sub_executor.submit(
                                lambda: df_without_gateway[
                                    df_without_gateway[meter_col_walkby].astype(str).isin(intermitente_walkby_meters)
                                ].copy()
                            )
                            tasks.append(('walkby_intermitente', task4))
                        
                        # Collect results
                        gateway_readings = []
                        walkby_readings = []
                        
                        for task_name, task in tasks:
                            try:
                                result = task.result()
                                if not result.empty:
                                    if task_name == 'gateway_diario':
                                        gateway_readings.append(result)
                                    else:  # All other tasks are walkby readings
                                        walkby_readings.append(result)
                            except Exception as e:
                                print(f"⚠️ Error en tarea {task_name}: {str(e)}")
                        
                        # Combine Gateway readings
                        if gateway_readings:
                            valorizable_gateway_df = pd.concat(gateway_readings, ignore_index=True)
                            
                            # Sort the Gateway data
                            if 'Record Time' in valorizable_gateway_df.columns and meter_col_gateway:
                                sort_cols = [meter_col_gateway, 'Record Time']
                            elif meter_col_gateway:
                                sort_cols = [meter_col_gateway]
                            else:
                                sort_cols = []
                            
                            if sort_cols:
                                valorizable_gateway_df = valorizable_gateway_df.sort_values(sort_cols).reset_index(drop=True)
                            
                            # Remove any duplicates
                            valorizable_gateway_df = valorizable_gateway_df.drop_duplicates().reset_index(drop=True)
                        
                        # Combine Walkby readings
                        if walkby_readings:
                            valorizable_walkby_df = pd.concat(walkby_readings, ignore_index=True)
                            
                            # Sort the Walkby data
                            if 'Record Time' in valorizable_walkby_df.columns and meter_col_walkby:
                                sort_cols = [meter_col_walkby, 'Record Time']
                            elif meter_col_walkby:
                                sort_cols = [meter_col_walkby]
                            else:
                                sort_cols = []
                            
                            if sort_cols:
                                valorizable_walkby_df = valorizable_walkby_df.sort_values(sort_cols).reset_index(drop=True)
                            
                            # Remove any duplicates
                            valorizable_walkby_df = valorizable_walkby_df.drop_duplicates().reset_index(drop=True)
            
            return valorizable_gateway_df, valorizable_walkby_df
        
        # Execute VALORIZABLE preparation in parallel - now returns two DataFrames
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            valorizable_future = executor.submit(prepare_valorizable_data)
            valorizable_gateway_df, valorizable_walkby_df = valorizable_future.result()
        
        # Consolidate valorizable data into a single DataFrame for the summary
        valorizable_df = pd.DataFrame()
        if not valorizable_gateway_df.empty:
            valorizable_df = pd.concat([valorizable_df, valorizable_gateway_df], ignore_index=True)
        if not valorizable_walkby_df.empty:
            valorizable_df = pd.concat([valorizable_df, valorizable_walkby_df], ignore_index=True)
        
        
        def create_summary_sheet(workbook, writer, pivot_table_df, valorizable_df, total_original_records, formats):
            """
            Create a summary sheet with charts showing valorizable data breakdown.
            This function now calculates everything based on the final pivot table.
            """
            summary_data = {
                'Tipo de Medidor': [],
                'Cantidad de Medidores': [],
                'Lecturas Valorizables': []
            }
            
            # Calculate all metrics in a single pass from the final pivot table
            if not pivot_table_df.empty:
                total_emrc_meters = len(pivot_table_df)
                
                # Use vectorized operations for counting
                clasificacion_counts = pivot_table_df['Clasificacion'].value_counts()
                incidencia_counts = pivot_table_df['Incidencia'].value_counts()
                
                # Extract counts efficiently
                sin_lectura_count = clasificacion_counts.get('SIN LECTURA', 0)
                con_incidencias_count = incidencia_counts.get('CON INCIDENCIA', 0)
                
                # Create combined condition mask for efficient filtering
                sin_incidencia_mask = pivot_table_df['Incidencia'] == 'SIN INCIDENCIA'
                
                diario_count = clasificacion_counts.get('DIARIO', 0)
                intermitente_count = clasificacion_counts.get('INTERMITENTE', 0)
                walkby_count = clasificacion_counts.get('WALKBY', 0)
                intermitente_walkby_count = clasificacion_counts.get('INTERMITENTE / WALKBY', 0)

                # Calculate readings for each type from the consolidated valorizable data
                if not valorizable_df.empty:
                    # Find meter column
                    meter_col = next((col for col in valorizable_df.columns if 'meter id' in col.lower() or 'meter_id' in col.lower()), None)
                    
                    if meter_col:
                        pivot_meter_col = next((col for col in pivot_table_df.columns if 'meter id' in col.lower() or 'meter_id' in col.lower()), None)

                        # Get lists of valorizable meters directly from the pivot table
                        diario_meters = set(pivot_table_df[(pivot_table_df['Clasificacion'] == 'DIARIO') & sin_incidencia_mask][pivot_meter_col].astype(str))
                        walkby_meters = set(pivot_table_df[(pivot_table_df['Clasificacion'] == 'WALKBY') & sin_incidencia_mask][pivot_meter_col].astype(str))
                        intermitente_walkby_meters = set(pivot_table_df[(pivot_table_df['Clasificacion'] == 'INTERMITENTE / WALKBY') & sin_incidencia_mask][pivot_meter_col].astype(str))
                        
                        # Convert valorizable meter column once for efficiency
                        valorizable_meters_series = valorizable_df[meter_col].astype(str)
                        
                        # Count readings for each type using vectorized isin operations
                        diario_readings = valorizable_meters_series[valorizable_meters_series.isin(diario_meters)].count()
                        walkby_readings = valorizable_meters_series[valorizable_meters_series.isin(walkby_meters)].count()
                        intermitente_walkby_readings = valorizable_meters_series[valorizable_meters_series.isin(intermitente_walkby_meters)].count()
                        intermitente_readings = 0  # Not valorizable
                    else:
                        diario_readings = walkby_readings = intermitente_walkby_readings = intermitente_readings = 0
                else:
                    diario_readings = walkby_readings = intermitente_walkby_readings = intermitente_readings = 0
            else:
                total_emrc_meters = sin_lectura_count = con_incidencias_count = 0
                diario_count = intermitente_count = walkby_count = intermitente_walkby_count = 0
                diario_readings = walkby_readings = intermitente_walkby_readings = intermitente_readings = 0

            # Populate summary data
            summary_data['Tipo de Medidor'] = [
                'DIARIO (Gateway)', 'INTERMITENTE', 'WALKBY (Manual)', 'INTERMITENTE / WALKBY', 'SIN LECTURA', 'CON INCIDENCIAS', 'TOTAL EMRC', 'TOTAL VALORIZABLES'
            ]
            summary_data['Cantidad de Medidores'] = [
                diario_count, intermitente_count, walkby_count, intermitente_walkby_count, sin_lectura_count, con_incidencias_count, total_emrc_meters,
                diario_count + walkby_count + intermitente_walkby_count
            ]
            summary_data['Lecturas Valorizables'] = [
                diario_readings, intermitente_readings, walkby_readings, intermitente_walkby_readings, 0, 0, 0,
                diario_readings + walkby_readings + intermitente_walkby_readings
            ]
            
            # Create DataFrame
            summary_df = pd.DataFrame(summary_data)
            
            # Write summary data to sheet
            summary_df.to_excel(writer, sheet_name='RESUMEN', index=False)
            worksheet = writer.sheets['RESUMEN']
            
            # Apply header formatting
            for col_num, value in enumerate(summary_df.columns.values):
                worksheet.write(0, col_num, value, formats['dark_blue'])
            worksheet.set_row(0, 25)
            
            # Apply data formatting - optimized with single pass
            for row_num in range(1, len(summary_df) + 1):
                row_type = summary_df.iloc[row_num - 1, 0]  # Get the "Tipo de Medidor" value once
                
                # Determine format once per row
                if row_type in ['TOTAL EMRC', 'TOTAL VALORIZABLES']:
                    row_format = formats['accent']
                elif row_type in ['SIN LECTURA', 'CON INCIDENCIAS']:
                    row_format = formats.get('yellow', formats['light_grey'])
                else:
                    row_format = formats['light_grey']
                
                # Apply format to entire row
                for col_num in range(len(summary_df.columns)):
                    value = summary_df.iloc[row_num - 1, col_num]
                    worksheet.write(row_num, col_num, value, row_format)
            
            # Set column widths
            worksheet.set_column(0, 0, 25)  # Tipo de Medidor
            worksheet.set_column(1, 1, 20)  # Cantidad de Medidores
            worksheet.set_column(2, 2, 20)  # Lecturas Valorizables
            
            # Create charts
            try:
                # Chart 1: Meters by Type (Pie Chart) - Only show valorizable meters
                chart1 = workbook.add_chart({'type': 'pie'})
                chart1.add_series({
                    'name': 'Medidores por Tipo',
                    'categories': ['RESUMEN', 1, 0, 4, 0],  # Now includes INTERMITENTE
                    'values': ['RESUMEN', 1, 1, 4, 1],
                    'data_labels': {
                        'value': True,
                        'percentage': True,
                        'separator': '\n',
                        'position': 'outside_end'
                    }
                })
                chart1.set_title({
                    'name': 'Distribución de Medidores Valorizables por Tipo',
                    'name_font': {'size': 14, 'bold': True}
                })
                chart1.set_size({'width': 500, 'height': 350})
                worksheet.insert_chart('E2', chart1)
                
                # Chart 2: Readings by Type (Column Chart) - Only show valorizable meters
                chart2 = workbook.add_chart({'type': 'column'})
                chart2.add_series({
                    'name': 'Lecturas Valorizables',
                    'categories': ['RESUMEN', 1, 0, 4, 0], # Now includes INTERMITENTE
                    'values': ['RESUMEN', 1, 2, 4, 2],
                    'fill': {'color': '#5B9BD5'},
                    'data_labels': {'value': True}
                })
                chart2.set_title({
                    'name': 'Lecturas Valorizables por Tipo de Medidor',
                    'name_font': {'size': 14, 'bold': True}
                })
                chart2.set_x_axis({
                    'name': 'Tipo de Medidor',
                    'name_font': {'size': 12, 'bold': True}
                })
                chart2.set_y_axis({
                    'name': 'Cantidad de Lecturas',
                    'name_font': {'size': 12, 'bold': True}
                })
                chart2.set_size({'width': 500, 'height': 350})
                worksheet.insert_chart('E20', chart2)
                
                # Add additional summary information
                worksheet.write(10, 0, 'INFORMACIÓN ADICIONAL:', formats['dark_blue'])
                worksheet.write(11, 0, f'Total registros originales: {total_original_records:,}', formats['light_grey'])
                worksheet.write(12, 0, f'Total lecturas valorizables: {len(valorizable_df):,}', formats['light_grey'])
                worksheet.write(13, 0, f'Total medidores EMRC: {total_emrc_meters:,}', formats['light_grey'])
                worksheet.write(14, 0, f'Medidores sin lectura: {sin_lectura_count:,}', formats.get('yellow', formats['light_grey']))
                worksheet.write(15, 0, f'Medidores con incidencias: {con_incidencias_count:,}', formats.get('yellow', formats['light_grey']))
                if total_original_records > 0:
                    percentage = (len(valorizable_df) / total_original_records) * 100
                    worksheet.write(16, 0, f'Porcentaje valorizable: {percentage:.1f}%', formats['light_grey'])
                if total_emrc_meters > 0:
                    valorizable_meters_count = diario_count + walkby_count + intermitente_walkby_count
                    emrc_percentage = (valorizable_meters_count / total_emrc_meters) * 100
                    worksheet.write(17, 0, f'Cobertura EMRC valorizable: {emrc_percentage:.1f}%', formats['light_grey'])
                
            except Exception as e:
                print(f"⚠️ Error creando gráficos en hoja resumen: {str(e)}")
                # Write error message
                worksheet.write(6, 0, f'Error creando gráficos: {str(e)}', formats['white'])
        
        def write_excel_parallel():
            """
            Optimized Excel writing with better performance and error handling
            """
            try:
                # Try xlsxwriter first for better performance and formatting
                with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
                    workbook = writer.book
                    
                    # Pre-define all formats for better performance
                    formats = {
                        'dark_blue': workbook.add_format({
                            'bold': True, 'text_wrap': True, 'valign': 'vcenter',
                            'fg_color': '#1F4E79', 'font_color': 'white', 'border': 1
                        }),
                        'light_blue': workbook.add_format({
                            'bold': True, 'text_wrap': True, 'valign': 'vcenter',
                            'fg_color': '#5B9BD5', 'font_color': 'white', 'border': 1
                        }),
                        'light_grey': workbook.add_format({
                            'bold': True, 'text_wrap': True, 'valign': 'vcenter',
                            'fg_color': '#D9D9D9', 'font_color': 'black', 'border': 1
                        }),
                        'white': workbook.add_format({
                            'bold': True, 'text_wrap': True, 'valign': 'vcenter',
                            'fg_color': '#FFFFFF', 'font_color': 'black', 'border': 1
                        }),
                        'accent': workbook.add_format({
                            'bold': True, 'text_wrap': True, 'valign': 'vcenter',
                            'fg_color': '#2F5F8F', 'font_color': 'white', 'border': 1
                        })
                    }
                    
                    # CREATE SUMMARY SHEET FIRST, now based on final results
                    create_summary_sheet(workbook, writer, pivot_table_df, valorizable_df, len(combined_df), formats)
                    
                    # Define sheet configurations for parallel processing
                    sheet_configs = [
                        ('TOTAL', combined_df, 'dark_blue', 20),
                        ('Gateway Depurado', df_with_gateway_limited, 'light_blue', 20),
                        ('WALKBY', df_without_gateway, 'light_grey', 20),
                        ('VALORIZABLE GATEWAY', valorizable_gateway_df, 'light_blue', 20),
                        ('VALORIZABLE WALKBY', valorizable_walkby_df, 'light_grey', 20)
                    ]
                    
                    # Write sheets efficiently
                    for sheet_name, df, format_name, row_height in sheet_configs:
                        if not df.empty:
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                            worksheet = writer.sheets[sheet_name]
                            
                            # Apply header formatting
                            for col_num, value in enumerate(df.columns.values):
                                worksheet.write(0, col_num, value, formats[format_name])
                            worksheet.set_row(0, row_height)
                        else:
                            # Create empty sheet with message for VALORIZABLE
                            if 'VALORIZABLE' in sheet_name:
                                empty_df = pd.DataFrame({'Mensaje': [f'No se encontraron lecturas para {sheet_name}']})
                                empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
                                worksheet = writer.sheets[sheet_name]
                                worksheet.write(0, 0, 'Mensaje', formats['white'])
                                worksheet.set_row(0, row_height)
                    
                    # Handle Eliminados sheet
                    if not deleted_df.empty:
                        deleted_df.to_excel(writer, sheet_name='Eliminados', index=False)
                        worksheet = writer.sheets['Eliminados']
                        for col_num, value in enumerate(deleted_df.columns.values):
                            worksheet.write(0, col_num, value, formats['white'])
                        worksheet.set_row(0, 20)
                    else:
                        empty_df = pd.DataFrame({'Mensaje': ['No se eliminaron lecturas']})
                        empty_df.to_excel(writer, sheet_name='Eliminados', index=False)
                        worksheet = writer.sheets['Eliminados']
                        worksheet.write(0, 0, 'Mensaje', formats['white'])
                        worksheet.set_row(0, 20)
                    
                    # Special handling for pivot table with enhanced formatting
                    if not pivot_table_df.empty:
                        pivot_table_df.to_excel(writer, sheet_name='Tabla Dinamica', index=False)
                        worksheet = writer.sheets['Tabla Dinamica']
                        
                        # Apply specialized formatting for different column types
                        for col_num, col_name in enumerate(pivot_table_df.columns.values):
                            if col_name in ['Meter ID', 'Customer ID']:
                                worksheet.write(0, col_num, col_name, formats['dark_blue'])
                            elif col_name in ['Clasificacion', 'Incidencia', 'Detalle de Incidencia']:
                                worksheet.write(0, col_num, col_name, formats['accent'])
                            elif col_name in ['Numero De Lecturas', 'Dias Lecturadas']:
                                worksheet.write(0, col_num, col_name, formats['light_blue'])
                            else:
                                worksheet.write(0, col_num, col_name, formats['light_grey'])
                        
                        worksheet.set_row(0, 25)
                        
                        # Auto-adjust column widths efficiently
                        for i, col in enumerate(pivot_table_df.columns):
                            try:
                                column_len = max(
                                    pivot_table_df[col].astype(str).map(len).max(),
                                    len(str(col))
                                ) + 2
                                worksheet.set_column(i, i, min(column_len, 30))
                            except:
                                worksheet.set_column(i, i, 15)  # Default width
                    else:
                        empty_df = pd.DataFrame({'Mensaje': ['No se pudo crear la tabla dinámica']})
                        empty_df.to_excel(writer, sheet_name='Tabla Dinamica', index=False)
                        worksheet = writer.sheets['Tabla Dinamica']
                        worksheet.write(0, 0, 'Mensaje', formats['white'])
                        worksheet.set_row(0, 20)
                
                return True
                
            except Exception as e:
                print(f"⚠️ Error con xlsxwriter, usando openpyxl: {str(e)}")
                # Fallback to openpyxl
                try:
                    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                        # Create a summary sheet first in fallback mode as well
                        if not pivot_table_df.empty:
                            # Simplified summary for fallback
                            summary_counts = pivot_table_df['Clasificacion'].value_counts().reset_index()
                            summary_counts.columns = ['Clasificacion', 'Cantidad de Medidores']
                            summary_counts.to_excel(writer, sheet_name='RESUMEN', index=False)

                        combined_df.to_excel(writer, sheet_name='TOTAL', index=False)
                        df_with_gateway_limited.to_excel(writer, sheet_name='Gateway Depurado', index=False)
                        df_without_gateway.to_excel(writer, sheet_name='WALKBY', index=False)
                        
                        if not deleted_df.empty:
                            deleted_df.to_excel(writer, sheet_name='Eliminados', index=False)
                        else:
                            pd.DataFrame({'Mensaje': ['No se eliminaron lecturas']}).to_excel(
                                writer, sheet_name='Eliminados', index=False
                            )
                        
                        if not valorizable_gateway_df.empty:
                            valorizable_gateway_df.to_excel(writer, sheet_name='VALORIZABLE GATEWAY', index=False)
                        else:
                            pd.DataFrame({'Mensaje': ['No se encontraron lecturas Gateway valorizables']}).to_excel(
                                writer, sheet_name='VALORIZABLE GATEWAY', index=False
                            )
                        
                        if not valorizable_walkby_df.empty:
                            valorizable_walkby_df.to_excel(writer, sheet_name='VALORIZABLE WALKBY', index=False)
                        else:
                            pd.DataFrame({'Mensaje': ['No se encontraron lecturas Walkby valorizables']}).to_excel(
                                writer, sheet_name='VALORIZABLE WALKBY', index=False
                            )
                        
                        if not pivot_table_df.empty:
                            pivot_table_df.to_excel(writer, sheet_name='Tabla Dinamica', index=False)
                        else:
                            pd.DataFrame({'Mensaje': ['No se pudo crear la tabla dinámica']}).to_excel(
                                writer, sheet_name='Tabla Dinamica', index=False
                            )
                    
                    return True
                    
                except Exception as e2:
                    print(f"❌ Error escribiendo archivo Excel: {str(e2)}")
                    return False
        
        # Execute Excel writing
        excel_success = write_excel_parallel()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Generate clean summary
        print("="*50)
        print("TABLA DINÁMICA:")
        
        # Count by classification
        if not pivot_table_df.empty:
            classification_counts = pivot_table_df['Clasificacion'].value_counts()
            for classification, count in classification_counts.items():
                print(f"  • {classification}: {count:,} medidores")
        
        print()
        print(f"  • Registros Gateway: {len(df_with_gateway_limited):,}")
        print(f"  • Registros WALKBY: {len(df_without_gateway):,}")
        if not valorizable_gateway_df.empty:
            print(f"  • Registros VALORIZABLE GATEWAY: {len(valorizable_gateway_df):,}")
        if not valorizable_walkby_df.empty:
            print(f"  • Registros VALORIZABLE WALKBY: {len(valorizable_walkby_df):,}")
        total_valorizable = len(valorizable_gateway_df) + len(valorizable_walkby_df)
        if total_valorizable > 0:
            print(f"  • Total registros VALORIZABLE: {total_valorizable:,}")
        print()
        
        if excel_success:
            print("✅ Proceso completado exitosamente")
        else:
            print("❌ Error durante el procesamiento")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error procesando archivo: {str(e)}")
        raise e

def main():
    # Get file paths using the updated function
    data_file_path, meters_file_path, incidents_file_path = get_file_paths()
    
    if not data_file_path:
        print("❌ No se proporcionó archivo de datos")
        return
    
    # Validate data file
    if not os.path.exists(data_file_path):
        print(f"❌ Archivo de datos no encontrado: {data_file_path}")
        return
    
    if not data_file_path.lower().endswith(('.xls', '.xlsx')):
        print("❌ El archivo de datos debe ser de Excel (.xls o .xlsx)")
        return
    
    # Validate meters file if provided
    if meters_file_path:
        if not os.path.exists(meters_file_path):
            print(f"⚠️ Archivo de medidores no encontrado: {meters_file_path}")
            print("Continuando sin archivo de medidores...")
            meters_file_path = None
        elif not meters_file_path.lower().endswith(('.xls', '.xlsx')):
            print("⚠️ El archivo de medidores debe ser de Excel (.xls o .xlsx)")
            print("Continuando sin archivo de medidores...")
            meters_file_path = None
    
    # Validate incidents file if provided
    if incidents_file_path:
        if not os.path.exists(incidents_file_path):
            print(f"⚠️ Archivo de incidencias no encontrado: {incidents_file_path}")
            print("Continuando sin archivo de incidencias...")
            incidents_file_path = None
        elif not incidents_file_path.lower().endswith(('.xls', '.xlsx')):
            print("⚠️ El archivo de incidencias debe ser de Excel (.xls o .xlsx)")
            print("Continuando sin archivo de incidencias...")
            incidents_file_path = None
    
    # Show file sizes
    data_file_size = os.path.getsize(data_file_path) / (1024 * 1024)  # MB
    print(f"📁 Archivo de datos: {data_file_size:.1f} MB")
    
    if meters_file_path:
        meters_file_size = os.path.getsize(meters_file_path) / (1024 * 1024)  # MB
        print(f"📁 Archivo de medidores: {meters_file_size:.1f} MB")
    
    if incidents_file_path:
        incidents_file_size = os.path.getsize(incidents_file_path) / (1024 * 1024)  # MB
        print(f"📁 Archivo de incidencias: {incidents_file_size:.1f} MB")
    
    # Process files using the updated function
    append_all_sheets_optimized(data_file_path, meters_file_path, incidents_file_path)

if __name__ == "__main__":
    main()