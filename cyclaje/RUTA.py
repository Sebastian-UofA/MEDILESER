import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import re

def get_file_path(title):
    """Opens a file dialog to select a file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=title)
    return file_path

def read_alarms_file(alarms_file_path):
    """
    Read the alarms file and extract alarm data for each meter
    Returns a dictionary: {meter_id: {alarm_type: count, ...}}
    """
    if not alarms_file_path or not os.path.exists(alarms_file_path):
        return {}
    
    try:
        # Read the file
        if alarms_file_path.endswith('.xls'):
            all_sheets = pd.read_excel(alarms_file_path, sheet_name=None, engine='xlrd')
        else:
            all_sheets = pd.read_excel(alarms_file_path, sheet_name=None, engine='openpyxl')
        
        # Use the first sheet
        first_sheet = next(iter(all_sheets.values()))
        
        if (first_sheet.empty):
            return {}
        
        alarms_dict = {}
        
        # Determine if first row is headers or data
        has_headers = False
        if len(first_sheet) > 0:
            first_row_values = first_sheet.iloc[0].tolist()
            first_cell = str(first_row_values[0]).strip().lower()
            # Support both English and Spanish header keywords
            header_keywords = [
                'meter', 'id', 'medidor', 'abnormal', 'flow', 'temperature', 'leakage', 'air', 'reverse', 'alarm', 'address',
                'tasa', 'flujo', 'temperatura', 'fugas', 'aire', 'tubo', 'reverso', 'alarma', 'dirección', 'direccion', 'cliente'
            ]
            
            if any(keyword in first_cell for keyword in header_keywords):
                has_headers = True
        
        start_row = 1 if has_headers else 0
        
        # Process data starting from appropriate row
        for idx in range(start_row, len(first_sheet)):
            try:
                row = first_sheet.iloc[idx]
                if len(row) >= 2 and pd.notna(row.iloc[0]):
                    meter_id = str(row.iloc[0]).strip()
                    
                    # Ensure meter ID has proper format (add 00 prefix if needed)
                    if meter_id.startswith(('KA', 'KB')) and not meter_id.startswith('00'):
                        meter_id = '00' + meter_id
                    
                    # Create alarm record
                    alarm_data = {}
                    
                    # Map alarm columns (these are counts of alarm occurrences)
                    if len(row) >= 2:
                        alarm_data['Abnormal Flow Rate'] = int(row.iloc[1]) if pd.notna(row.iloc[1]) and str(row.iloc[1]).strip() != '' else 0
                    if len(row) >= 3:
                        alarm_data['Abnormal Temperature'] = int(row.iloc[2]) if pd.notna(row.iloc[2]) and str(row.iloc[2]).strip() != '' else 0
                    if len(row) >= 4:
                        alarm_data['Leakage'] = int(row.iloc[3]) if pd.notna(row.iloc[3]) and str(row.iloc[3]).strip() != '' else 0
                    if len(row) >= 5:
                        alarm_data['Air In Pipe'] = int(row.iloc[4]) if pd.notna(row.iloc[4]) and str(row.iloc[4]).strip() != '' else 0
                    if len(row) >= 6:
                        alarm_data['Reverse Flow'] = int(row.iloc[5]) if pd.notna(row.iloc[5]) and str(row.iloc[5]).strip() != '' else 0
                    
                    alarms_dict[meter_id] = alarm_data
                    
            except (IndexError, ValueError, KeyError):
                continue
        
        return alarms_dict
        
    except Exception as e:
        print(f"Error reading alarms file: {e}")
        return {}

def determinar_analisis(row, last_two_days):
    """Determines the analysis status for a meter based on incidents, readings, and alarms."""
    # Priority 1: Based on Incidents File
    if 'OBSERVADO / INCIDENCIA' in row:
        status = row['OBSERVADO / INCIDENCIA']
        if status == 'OBSERVADO':
            return 'RUTA RECOMENDADA'
        if status == 'INCIDENCIA':
            return 'NO ENVIADO'  # Skipped as per instructions

    # Priority 2: Based on Reading Data
    total_readings = row.get('Total Readings', 0)
    
    # Condition: Zero total readings for the entire period
    if total_readings == 0:
        return 'RUTA RECOMENDADA'
    
    # Condition: Zero readings in the last two days
    if last_two_days:
        readings_last_two_days = sum(row.get(day, 0) for day in last_two_days)
        if readings_last_two_days == 0:
            return 'RUTA RECOMENDADA'

    # Priority 3: Based on Alarms Data
    # INSPECCION for Air In Pipe or Reverse Flow
    inspeccion_alarms = ['Air In Pipe', 'Reverse Flow']
    if total_readings > 0:
        for alarm in inspeccion_alarms:
            if row.get(alarm, 0) > (total_readings / 2):
                return 'INSPECCION'
            
    # AVISO for other alarms
    aviso_alarms = ['Abnormal Flow Rate', 'Abnormal Temperature', 'Leakage']
    if total_readings > 0:
        for alarm in aviso_alarms:
            if row.get(alarm, 0) > (total_readings / 2):
                return 'AVISO'

    # Default case if no other condition is met
    return 'NO ENVIADO'

def main():
    """
    Main function to process water meter data.
    """
    # Get input files from user
    meter_ids_file = get_file_path("Seleccione el archivo de texto de los medidores")
    if not meter_ids_file:
        print("No meter ID file selected. Exiting.")
        return

    readings_file = get_file_path("Seleccione el archivo Excel con las lecturas de los medidores")
    if not readings_file:
        print("No readings file selected. Exiting.")
        return

    incident_file = get_file_path("Seleccione el archivo de incidentes (opcional)")
    alarms_file = get_file_path("Seleccione el archivo de alarmas (opcional)")

    # Read incident file if provided
    df_incidents = pd.DataFrame()
    if incident_file:
        try:
            # The incident file is an Excel file.
            engine = 'xlrd' if incident_file.lower().endswith('.xls') else 'openpyxl'
            df_incidents = pd.read_excel(incident_file, engine=engine, header=None, names=['Meter ID', 'DETALLE', 'OBSERVADO / INCIDENCIA'])
            # Normalize meter IDs in the incident file
            df_incidents['Meter ID'] = df_incidents['Meter ID'].astype(str).str.strip()
            df_incidents['Meter ID'] = df_incidents['Meter ID'].apply(lambda mid: f"00{mid}" if mid.startswith(('KA', 'KB')) and not mid.startswith('00') else mid)
            df_incidents.set_index('Meter ID', inplace=True)
        except Exception as e:
            print(f"Could not read or process incident file: {e}")
            df_incidents = pd.DataFrame()

    # Read alarms file if provided
    df_alarms = pd.DataFrame()
    if alarms_file:
        alarms_data = read_alarms_file(alarms_file)
        if alarms_data:
            df_alarms = pd.DataFrame.from_dict(alarms_data, orient='index')

    # Read meter IDs from the text file 
    meter_ids_raw = []
    meter_id_pattern = re.compile(r'^(KA|KB)\d{8}$')
    try:
        with open(meter_ids_file, 'r', encoding='latin-1') as f:
            for line in f:
                # Split by tab, as it seems to be the main delimiter
                parts = line.strip().split('\t')
                # Search for the meter ID in the parts of the line
                for part in parts:
                    cleaned_part = part.strip()
                    if meter_id_pattern.match(cleaned_part):
                        meter_ids_raw.append(cleaned_part)
                        # Assuming one meter ID per line, we can stop after finding one
                        break 
    except Exception as e:
        print(f"Error reading or parsing meter ID file: {e}")
        return
    
    # Normalize meter IDs (add '00' prefix if needed)
    meter_ids = [f"00{mid}" if mid.startswith(('KA', 'KB')) and not mid.startswith('00') else mid for mid in meter_ids_raw]
    meter_ids_set = set(meter_ids)

    # Read all sheets from the Excel file and combine them
    try:
        # Determine the correct engine based on file extension
        engine = 'xlrd' if readings_file.lower().endswith('.xls') else 'openpyxl'
        
        # Read all sheets into a dictionary of DataFrames
        all_sheets = pd.read_excel(readings_file, sheet_name=None, engine=engine)
        
        # Concatenate all DataFrames from the sheets into a single DataFrame
        df = pd.concat(all_sheets.values(), ignore_index=True)

    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Process 'Record Time'
    if 'Record Time' in df.columns:
        df['Record Time'] = pd.to_datetime(df['Record Time'])
        df.insert(df.columns.get_loc('Record Time') + 1, 'Date', df['Record Time'].dt.date)
        df.insert(df.columns.get_loc('Date') + 1, 'Time', df['Record Time'].dt.time)
    else:
        print("Column 'Record Time' not found in the Excel file.")
        return

    # Filter dataframe based on meter IDs
    df_filtered = df[df['Meter ID'].isin(meter_ids_set)].copy()

    # Create pivot table
    if not df_filtered.empty:
        df_filtered['DateOnly'] = pd.to_datetime(df_filtered['Date']).dt.date
        pivot_table = pd.pivot_table(df_filtered, 
                                     index='Meter ID', 
                                     columns='DateOnly', 
                                     aggfunc='size', 
                                     fill_value=0)
        date_columns = sorted([col for col in pivot_table.columns if hasattr(col, 'year')])
    else:
        # If no readings found for any of the meters, create an empty DataFrame
        pivot_table = pd.DataFrame()
        date_columns = []

    # Ensure all meter IDs from the input file are in the pivot table's index
    pivot_table = pivot_table.reindex(sorted(list(meter_ids_set))).fillna(0).astype(int)

    # Add summary columns to pivot table
    pivot_table['Total Readings'] = pivot_table.sum(axis=1)
    pivot_table['Days with Readings'] = (pivot_table.iloc[:, :-1] > 0).sum(axis=1)

    # Merge with incident data if available
    if not df_incidents.empty:
        pivot_table = pivot_table.join(df_incidents, how='left')
        pivot_table[['DETALLE', 'OBSERVADO / INCIDENCIA']] = pivot_table[['DETALLE', 'OBSERVADO / INCIDENCIA']].fillna('SIN INCIDENCIA')

    # Merge with alarm data if available
    if not df_alarms.empty:
        pivot_table = pivot_table.join(df_alarms, how='left')
        # Fill NaN values for alarm columns with 0 for counts
        alarm_cols = ['Abnormal Flow Rate', 'Abnormal Temperature', 'Leakage', 'Air In Pipe', 'Reverse Flow']
        for col in alarm_cols:
            if col in pivot_table.columns:
                pivot_table[col] = pivot_table[col].fillna(0).astype(int)

    # Add 'Análisis' column based on the specified logic
    last_two_days = date_columns[-2:] if len(date_columns) >= 2 else date_columns
    pivot_table['Análisis'] = pivot_table.apply(lambda row: determinar_analisis(row, last_two_days), axis=1)

    # --- Summary Table Calculation ---
    total_meters = len(pivot_table)
    ruta_count = (pivot_table['Análisis'] == 'RUTA RECOMENDADA').sum()
    inspeccion_count = (pivot_table['Análisis'] == 'INSPECCION').sum()
    aviso_count = (pivot_table['Análisis'] == 'AVISO').sum()
    sin_recomendacion_count = total_meters - ruta_count - inspeccion_count - aviso_count

    summary_data = {
        'Categoría': ['Total de Medidores Analizados', 'Ruta Recomendada', 'Inspeccion', 'Aviso', 'Sin Recomendación'],
        'Cantidad': [total_meters, ruta_count, inspeccion_count, aviso_count, sin_recomendacion_count]
    }
    summary_df = pd.DataFrame(summary_data)

    # Prepare for output
    output_filename = os.path.join(os.path.dirname(readings_file), 'Valorization_Summary.xlsx')
    
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        # Write processed data
        sheet_name = f'Total Readings - {len(df)}'
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Write pivot table
        pivot_sheet_name = 'Meter Reading Pivot Table'
        pivot_table.to_excel(writer, sheet_name=pivot_sheet_name)

        # --- Formatting ---
        workbook = writer.book
        worksheet = writer.sheets[pivot_sheet_name]

        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        date_header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1,
            'num_format': 'yyyy-mm-dd'
        })
        ruta_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        aviso_format = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
        inspeccion_format = workbook.add_format({'bg_color': '#FFC000', 'font_color': '#9C5700'}) # Orange

        # Apply header format
        import datetime
        for col_num, value in enumerate(pivot_table.reset_index().columns.values):
            if isinstance(value, datetime.date):
                worksheet.write(0, col_num, value, date_header_format)
            else:
                worksheet.write(0, col_num, value, header_format)

        # Find the 'Análisis' column index
        try:
            analisis_col_idx = pivot_table.reset_index().columns.get_loc('Análisis')
            
            # Apply conditional formatting
            worksheet.conditional_format(1, analisis_col_idx, len(pivot_table), analisis_col_idx, {
                'type': 'cell',
                'criteria': '==',
                'value': '"RUTA RECOMENDADA"',
                'format': ruta_format
            })
            worksheet.conditional_format(1, analisis_col_idx, len(pivot_table), analisis_col_idx, {
                'type': 'cell',
                'criteria': '==',
                'value': '"AVISO"',
                'format': aviso_format
            })
            worksheet.conditional_format(1, analisis_col_idx, len(pivot_table), analisis_col_idx, {
                'type': 'cell',
                'criteria': '==',
                'value': '"INSPECCION"',
                'format': inspeccion_format
            })
        except KeyError:
            # 'Análisis' column might not exist if there's no data
            pass

        # Auto-fit columns for the pivot table
        for i, col in enumerate(pivot_table.reset_index().columns):
            # Convert column header to string to handle date objects
            header_len = len(str(col))
            # Get max length of data in the column
            data_len = pivot_table.reset_index()[col].astype(str).map(len).max()
            # Set column width
            column_len = max(data_len, header_len)
            worksheet.set_column(i, i, column_len + 2)

        # --- Write Summary Table ---
        summary_start_col = len(pivot_table.columns) + 2  # Position it to the right
        worksheet.write(1, summary_start_col, "Resumen del Análisis", header_format)
        
        for col_num, value in enumerate(summary_df.columns.values):
            worksheet.write(2, summary_start_col + col_num, value, header_format)

        for row_num, data_row in summary_df.iterrows():
            for col_num, cell_value in enumerate(data_row):
                worksheet.write(row_num + 3, summary_start_col + col_num, cell_value)

        for i, col in enumerate(summary_df.columns):
            column_len = max(summary_df[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(summary_start_col + i, summary_start_col + i, column_len + 2)

    print(f"Processing complete. Output saved to {output_filename}")

if __name__ == "__main__":
    main()
