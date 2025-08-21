import math
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import datetime

def get_file_path(title):
    """Opens a file dialog to select a file."""
    root = tk.Tk()
    root.withdraw()
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
        if alarms_file_path.endswith('.xls'):
            all_sheets = pd.read_excel(alarms_file_path, sheet_name=None, engine='xlrd')
        else:
            all_sheets = pd.read_excel(alarms_file_path, sheet_name=None, engine='openpyxl')
        
        first_sheet = next(iter(all_sheets.values()))
        
        if first_sheet.empty:
            return {}
        
        alarms_dict = {}
        
        has_headers = False
        if len(first_sheet) > 0:
            first_row_values = first_sheet.iloc[0].tolist()
            first_cell = str(first_row_values[0]).strip().lower()
            header_keywords = [
                'meter', 'id', 'medidor', 'abnormal', 'flow', 'temperature', 'leakage', 'air', 'reverse', 'alarm', 'address',
                'tasa', 'flujo', 'temperatura', 'fugas', 'aire', 'tubo', 'reverso', 'alarma', 'dirección', 'direccion', 'cliente'
            ]
            
            if any(keyword in first_cell for keyword in header_keywords):
                has_headers = True
        
        start_row = 1 if has_headers else 0
        
        for idx in range(start_row, len(first_sheet)):
            try:
                row = first_sheet.iloc[idx]
                if len(row) >= 2 and pd.notna(row.iloc[0]):
                    meter_id = str(row.iloc[0]).strip()
                    
                    if meter_id.startswith(('KA', 'KB')) and not meter_id.startswith('00'):
                        meter_id = '00' + meter_id
                    
                    alarm_data = {}
                    
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

def determinar_status(row, second_last_day_col, date_cols):
    """Determines the status based on incidents and readings."""
    # Priority 1: Incident
    if 'OBSERVADO / INCIDENCIA' in row and row['OBSERVADO / INCIDENCIA'] == 'INCIDENCIA':
        return 'INCIDENTE'

    # Priority 2: No data in any reading column
    if row[date_cols].isnull().all():
        return 'NO LECTURA'

    # Priority 3: No reading on the second to last day
    # Check if the column for the second to last day exists and if the value is NaN
    if second_last_day_col not in row.index or pd.isna(row[second_last_day_col]):
        return 'NO LECTURA'

    # Priority 4: Reading exists
    return 'OK'


def calculate_suggested_reading(row, date_cols, df_total_readings):
    """Calculates the suggested reading based on specific logic."""
    # If status is 'INCIDENTE', return 'INCIDENTE'
    if row['status'] == 'INCIDENTE':
        return 'INCIDENTE'

    num_days = len(date_cols)
    if num_days < 2:
        return ''  # Not enough data

    second_last_day_col = date_cols[-2]

    # Condition: No reading on the second-to-last day
    if pd.isna(row[second_last_day_col]):
        meter_id = row.name
        meter_readings = df_total_readings[df_total_readings['Meter ID'] == meter_id]

        # Find last reading of the third-to-last day
        val1 = None
        if num_days >= 3:
            third_last_day = date_cols[-3]
            third_day_readings = meter_readings[meter_readings['DateOnly'] == third_last_day]
            if not third_day_readings.empty:
                val1 = third_day_readings.sort_values('Record Time', ascending=False).iloc[0]['TotalFlow']

        # Find first reading of the last day
        val2 = None
        last_day = date_cols[-1]
        last_day_readings = meter_readings[meter_readings['DateOnly'] == last_day]
        if not last_day_readings.empty:
            val2 = last_day_readings.sort_values('Record Time', ascending=True).iloc[0]['TotalFlow']

        # Calculate suggested reading
        if pd.notna(val1) and pd.notna(val2):
            return math.floor((val1 + val2) / 2)
        elif pd.notna(val1):
            return val1
        elif pd.notna(val2):
            return val2
        else:
            # If no values for average, find the last known reading
            all_meter_readings = row[date_cols].dropna()
            if not all_meter_readings.empty:
                return all_meter_readings.iloc[-1]
            else:
                return 'NO INFO'
    
    return '' # Default empty value if condition is not met


def main():
    """
    Main function to process water meter data and generate a report.
    """
    # 1. Get input files from user
    readings_file = get_file_path("Select the Excel file with meter readings")
    if not readings_file:
        print("No readings file selected. Exiting.")
        return

    valorization_summary_file = get_file_path("Select the Valorization_Summary.xlsx file from cyclaje.py")
    if not valorization_summary_file:
        print("No Valorization Summary file selected. Exiting.")
        return

    incident_file = get_file_path("Select the updated incident file (optional)")
    alarms_file = get_file_path("Select the updated alarms file (optional)")

    # Read incident file if provided
    df_incidents = pd.DataFrame()
    if incident_file:
        try:
            engine = 'xlrd' if incident_file.lower().endswith('.xls') else 'openpyxl'
            df_incidents = pd.read_excel(incident_file, engine=engine, header=None, names=['Meter ID', 'DETALLE', 'OBSERVADO / INCIDENCIA'])
            df_incidents['Meter ID'] = df_incidents['Meter ID'].astype(str).str.strip()
            df_incidents['Meter ID'] = df_incidents['Meter ID'].apply(lambda mid: f"00{mid}" if mid.startswith(('KA', 'KB')) and not mid.startswith('00') else mid)
            df_incidents.set_index('Meter ID', inplace=True)
        except Exception as e:
            print(f"Could not read or process incident file: {e}")

    # Read alarms file if provided
    df_alarms = pd.DataFrame()
    if alarms_file:
        alarms_data = read_alarms_file(alarms_file)
        if alarms_data:
            df_alarms = pd.DataFrame.from_dict(alarms_data, orient='index')

    # 2. Read and parse data readings like in cyclaje.py
    try:
        engine = 'xlrd' if readings_file.lower().endswith('.xls') else 'openpyxl'
        all_sheets = pd.read_excel(readings_file, sheet_name=None, engine=engine)
        df_total_readings = pd.concat(all_sheets.values(), ignore_index=True)
    except Exception as e:
        print(f"Error reading the readings file: {e}")
        return

    # 3. Read the pivot table from the Valorization_Summary.xlsx file
    try:
        df_pivot_table = pd.read_excel(valorization_summary_file, sheet_name=1, index_col=0)
        meters_in_original_pivot = df_pivot_table.index
    except Exception as e:
        print(f"Error reading the Valorization_Summary.xlsx file: {e}")
        return

    # Filter the total readings to only include meters from the original pivot table
    df_total_readings = df_total_readings[df_total_readings['Meter ID'].isin(meters_in_original_pivot)]

    # 4. Create the new pivot table for max/min flow
    if 'Record Time' in df_total_readings.columns:
        df_total_readings['Record Time'] = pd.to_datetime(df_total_readings['Record Time'])
        df_total_readings['DateOnly'] = df_total_readings['Record Time'].dt.date
    else:
        print("Column 'Record Time' not found in the total readings sheet.")
        return
        
    if 'TotalFlow' in df_total_readings.columns:
        df_total_readings['TotalFlow'] = pd.to_numeric(df_total_readings['TotalFlow'], errors='coerce')
    else:
        print("Column 'TotalFlow' not found in the total readings sheet.")
        return

    # Create the pivot table for maximum flow
    flow_pivot_max = pd.pivot_table(df_total_readings,
                                    index='Meter ID',
                                    columns='DateOnly',
                                    values='TotalFlow',
                                    aggfunc='max')

    # Ensure all meters from the original pivot are present
    flow_pivot_max = flow_pivot_max.reindex(meters_in_original_pivot)

    second_last_day_col = None
    if not flow_pivot_max.empty:
        if len(flow_pivot_max.columns) > 1:
            second_last_day_col = flow_pivot_max.columns[-2]

        last_day = flow_pivot_max.columns[-1]
        flow_pivot_min_last_day = pd.pivot_table(df_total_readings[df_total_readings['DateOnly'] == last_day],
                                                 index='Meter ID',
                                                 columns='DateOnly',
                                                 values='TotalFlow',
                                                 aggfunc='min')
        # Update the last day's column in the max pivot table with the minimum values
        if not flow_pivot_min_last_day.empty:
            flow_pivot_max[last_day] = flow_pivot_min_last_day[last_day]

    # Merge with incident data if available
    if not df_incidents.empty:
        flow_pivot_max = flow_pivot_max.join(df_incidents, how='left')
        flow_pivot_max[['DETALLE', 'OBSERVADO / INCIDENCIA']] = flow_pivot_max[['DETALLE', 'OBSERVADO / INCIDENCIA']].fillna('SIN INCIDENCIA')

    # Merge with alarm data if available
    if not df_alarms.empty:
        flow_pivot_max = flow_pivot_max.join(df_alarms, how='left')
        alarm_cols = ['Abnormal Flow Rate', 'Abnormal Temperature', 'Leakage', 'Air In Pipe', 'Reverse Flow']
        for col in alarm_cols:
            if col in flow_pivot_max.columns:
                flow_pivot_max[col] = flow_pivot_max[col].fillna(0).astype(int)

    # Add 'RUTA RECOMENDADA' column based on original pivot table's 'Análisis'
    if 'Análisis' in df_pivot_table.columns:
        analisis_series = df_pivot_table['Análisis']
        flow_pivot_max = flow_pivot_max.join(analisis_series.rename('Análisis_Original'), how='left')
        flow_pivot_max['RUTA RECOMENDADA'] = flow_pivot_max['Análisis_Original'].apply(
            lambda x: 'RUTA RECOMENDADA' if pd.notna(x) and x == 'RUTA RECOMENDADA' else ''
        )
        flow_pivot_max.drop(columns=['Análisis_Original'], inplace=True)

    # Add 'status' column
    date_cols = [col for col in flow_pivot_max.columns if isinstance(col, datetime.date)]
    if len(date_cols) > 1:
        second_last_day_col = date_cols[-2]
        flow_pivot_max['status'] = flow_pivot_max.apply(lambda row: determinar_status(row, second_last_day_col, date_cols), axis=1)
        # Add 'Suggested Reading' column
        flow_pivot_max['Suggested Reading'] = flow_pivot_max.apply(lambda row: calculate_suggested_reading(row, date_cols, df_total_readings), axis=1)
    else:
        # Handle case with less than 2 days of data
        flow_pivot_max['status'] = 'N/A'
        flow_pivot_max['Suggested Reading'] = ''

    # 5. Prepare for output
    output_filename = os.path.join(os.path.dirname(readings_file), 'Informe_Ciclos.xlsx')

    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        df_total_readings.to_excel(writer, sheet_name='Total Readings', index=False)
        df_pivot_table.to_excel(writer, sheet_name='Meter Reading Pivot Table')

        # Sheet 3: Max/Min Flow Pivot Table
        if not flow_pivot_max.empty:
            flow_pivot_max.to_excel(writer, sheet_name='Max-Min Flow Pivot Table')

        # --- Formatting ---
        workbook = writer.book
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top', 
            'fg_color': '#D7E4BC', 'border': 1
        })
        date_header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top', 
            'fg_color': '#D7E4BC', 'border': 1, 'num_format': 'yyyy-mm-dd'
        })
        no_lectura_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        max_flow_header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top',
            'fg_color': '#C5D9F1', 'border': 1, 'num_format': 'yyyy-mm-dd'  # Light Blue
        })
        min_flow_header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top',
            'fg_color': '#D8E4BC', 'border': 1, 'num_format': 'yyyy-mm-dd'  # Light Green
        })
        suggested_reading_format = workbook.add_format({'bg_color': '#FFFF99'})  # Light Yellow
        no_info_format = workbook.add_format({'bg_color': '#FFC000', 'font_color': '#000000'})  # Orange for NO INFO

        # Format Total Readings sheet
        worksheet_total = writer.sheets['Total Readings']
        for col_num, value in enumerate(df_total_readings.columns.values):
            worksheet_total.write(0, col_num, value, header_format)
        for i, col in enumerate(df_total_readings.columns):
            column_len = max(df_total_readings[col].astype(str).map(len).max(), len(col))
            worksheet_total.set_column(i, i, column_len + 2)

        
        worksheet_pivot = writer.sheets['Meter Reading Pivot Table']
        for col_num, value in enumerate(df_pivot_table.reset_index().columns.values):
            if isinstance(value, datetime.date):
                worksheet_pivot.write(0, col_num, value, date_header_format)
            else:
                worksheet_pivot.write(0, col_num, value, header_format)
        for i, col in enumerate(df_pivot_table.reset_index().columns):
            header_len = len(str(col))
            data_len = df_pivot_table.reset_index()[col].astype(str).map(len).max()
            column_len = max(data_len, header_len)
            worksheet_pivot.set_column(i, i, column_len + 2)

        # Format Max-Min Flow Pivot Table sheet
        if not flow_pivot_max.empty:
            worksheet_flow = writer.sheets['Max-Min Flow Pivot Table']
            
            # Get date columns to identify max/min flow columns
            date_cols_from_pivot = [col for col in flow_pivot_max.columns if isinstance(col, datetime.date)]
            last_date_col = date_cols_from_pivot[-1] if date_cols_from_pivot else None

            for col_num, value in enumerate(flow_pivot_max.reset_index().columns.values):
                if isinstance(value, datetime.date):
                    worksheet_flow.write(0, col_num, value, max_flow_header_format)
                else:
                    worksheet_flow.write(0, col_num, value, header_format)
            
            # Add conditional formatting for 'status' column
            if 'status' in flow_pivot_max.reset_index().columns:
                status_col_idx = flow_pivot_max.reset_index().columns.get_loc('status')
                worksheet_flow.conditional_format(1, status_col_idx, len(flow_pivot_max), status_col_idx, {
                    'type': 'cell',
                    'criteria': '==',
                    'value': '"NO LECTURA"',
                    'format': no_lectura_format
                })

            # Add conditional formatting for 'Suggested Reading' column
            if 'Suggested Reading' in flow_pivot_max.reset_index().columns:
                suggested_col_idx = flow_pivot_max.reset_index().columns.get_loc('Suggested Reading')
                # Rule 1: Format non-blank cells with yellow
                worksheet_flow.conditional_format(1, suggested_col_idx, len(flow_pivot_max), suggested_col_idx, {
                    'type': 'no_blanks',
                    'format': suggested_reading_format
                })
                # Rule 2: Format 'NO INFO' with grey, overriding the yellow
                worksheet_flow.conditional_format(1, suggested_col_idx, len(flow_pivot_max), suggested_col_idx, {
                    'type': 'cell',
                    'criteria': '==',
                    'value': '"NO INFO"',
                    'format': no_info_format
                })

            for i, col in enumerate(flow_pivot_max.reset_index().columns):
                header_len = len(str(col))
                data_len = flow_pivot_max.reset_index()[col].astype(str).map(len).max()
                column_len = max(data_len, header_len)
                worksheet_flow.set_column(i, i, column_len + 2)

    print(f"Processing complete. Output saved to {output_filename}")

if __name__ == "__main__":
    main()
