"""
Fast Parallel Excel Data Extractor
This version uses multiprocessing for faster file processing
"""

import os
import pandas as pd
import re
from pathlib import Path
import logging
from datetime import datetime
from config import CONFIG, MONTHS_SPANISH
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Set up logging
logging.basicConfig(level=getattr(logging, CONFIG['LOG_LEVEL']), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_data_from_single_file(file_info):
    """
    Function to extract data from a single Excel file
    This will be called in parallel for each file
    """
    try:
        file_path = file_info['file_path']
        
        # Quick Excel processing
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        # Find DATOS sheet
        datos_sheet = None
        target_sheet_name = CONFIG['TARGET_SHEET_NAME'].upper()
        
        for sheet in sheet_names:
            if sheet.upper() == target_sheet_name:
                datos_sheet = sheet
                break
        
        if not datos_sheet:
            return None
        
        # Read only first 2 columns and 50 rows for speed
        df = pd.read_excel(file_path, sheet_name=datos_sheet, header=None, 
                         usecols=[0, 1], nrows=50)
        
        # Extract data using config patterns
        extracted_data = {}
        for field_key in CONFIG['DATA_FIELDS'].keys():
            extracted_data[field_key] = None
        
        for idx, row in df.iterrows():
            if len(row) >= 2:
                dato = str(row[0]).strip().upper() if pd.notna(row[0]) else ""
                valor = row[1] if pd.notna(row[1]) else None
                
                for field_key, keywords in CONFIG['DATA_FIELDS'].items():
                    if extracted_data[field_key] is None:
                        if all(keyword.upper() in dato for keyword in keywords):
                            extracted_data[field_key] = valor
                            break
        
        # Return complete record
        return {
            'Proyecto': file_info['proyecto'],
            'Año': file_info['year'],
            'Mes': file_info['month'],
            'Equipo_Comercial': file_info['equipo_comercial'],
            'Lecturas': extracted_data.get('lecturas'),
            'Cantidad_Medidores': extracted_data.get('cantidad_medidores'),
            'Valorizable_Gateway': extracted_data.get('valorizable_gateway'),
            'Valorizable_Walkby': extracted_data.get('valorizable_walkby'),
            'File_Path': str(file_path),
            'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        print(f"Error processing {file_info['file_path']}: {str(e)}")
        return None

class FastExcelExtractor:
    def __init__(self):
        self.config = CONFIG
        self.base_path = Path(CONFIG['BASE_PATH'])
        self.master_file_path = Path(CONFIG['MASTER_FILE_PATH'])
        
    def extract_year(self, folder_name):
        """Extract year from folder name"""
        match = re.search(r'\b(\d{4})\b', folder_name)
        return match.group(1) if match else None
    
    def extract_month(self, folder_name):
        """Extract month from folder name"""
        parts = folder_name.split('.')
        if len(parts) > 1:
            month_part = parts[1].strip().split()[0]
            return month_part.upper()
        return None
    
    def find_all_resumen_files(self, proyecto_name=None):
        """Find all RESUMEN files quickly"""
        if not self.base_path.exists():
            logger.error(f"Base path does not exist: {self.base_path}")
            return []
        
        found_files = []
        
        # Get proyecto folders
        if proyecto_name:
            proyecto_paths = [self.base_path / proyecto_name]
        else:
            proyecto_paths = [p for p in self.base_path.iterdir() if p.is_dir()]
        
        for proyecto_path in proyecto_paths:
            if not proyecto_path.exists():
                continue
                
            logger.info(f"Scanning proyecto: {proyecto_path.name}")
            
            # Scan directory structure efficiently
            try:
                for year_folder in proyecto_path.iterdir():
                    if not year_folder.is_dir():
                        continue
                    if not re.search(self.config['YEAR_FOLDER_PATTERN'], year_folder.name, re.IGNORECASE):
                        continue
                        
                    year = self.extract_year(year_folder.name)
                    
                    for month_folder in year_folder.iterdir():
                        if not month_folder.is_dir():
                            continue
                            
                        month = self.extract_month(month_folder.name)
                        
                        for equipo_folder in month_folder.iterdir():
                            if not equipo_folder.is_dir():
                                continue
                                
                            equipo_comercial = equipo_folder.name
                            
                            # Find RESUMEN files
                            for file in equipo_folder.iterdir():
                                if (file.is_file() and 
                                    file.name.upper().startswith(self.config['RESUMEN_FILE_PREFIX'].upper()) and 
                                    file.suffix.lower() in self.config['EXCEL_EXTENSIONS']):
                                    
                                    found_files.append({
                                        'file_path': file,
                                        'proyecto': proyecto_path.name,
                                        'year': year,
                                        'month': month,
                                        'equipo_comercial': equipo_comercial
                                    })
            
            except Exception as e:
                logger.error(f"Error scanning {proyecto_path}: {str(e)}")
        
        return found_files
    
    def process_files_parallel(self, files_info, max_workers=None):
        """Process files in parallel for maximum speed"""
        if not files_info:
            logger.warning("No files to process")
            return []
        
        # Use optimal number of workers
        if max_workers is None:
            max_workers = min(len(files_info), mp.cpu_count())
        
        logger.info(f"Processing {len(files_info)} files with {max_workers} workers...")
        
        all_records = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            future_to_file = {executor.submit(extract_data_from_single_file, file_info): file_info 
                            for file_info in files_info}
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        all_records.append(result)
                        # Reduced logging - only show progress occasionally
                        if len(all_records) % 5 == 0 or len(all_records) == len(files_info):
                            print(f"Processed {len(all_records)}/{len(files_info)} files...")
                    else:
                        logger.debug(f"No data extracted from: {file_info['file_path']}")
                except Exception as e:
                    logger.warning(f"Error processing {file_info['file_path']}: {str(e)}")
        
        return all_records
    
    def save_results(self, records):
        """Save results to master Excel file - update existing or create new"""
        if not records:
            logger.warning("No records to save")
            return
        
        try:
            new_df = pd.DataFrame(records)
            
            # Check if master file already exists
            if self.master_file_path.exists():
                print("Master file exists - checking for updates...")
                
                # Read existing data
                try:
                    existing_df = pd.read_excel(self.master_file_path, sheet_name='Extracted_Data')
                    print(f"Found {len(existing_df)} existing records")
                    
                    # Create unique identifiers for comparison
                    # Using Proyecto, Año, Mes, Equipo_Comercial as unique key
                    existing_df['unique_key'] = existing_df['Proyecto'].astype(str) + "_" + \
                                              existing_df['Año'].astype(str) + "_" + \
                                              existing_df['Mes'].astype(str) + "_" + \
                                              existing_df['Equipo_Comercial'].astype(str)
                    
                    new_df['unique_key'] = new_df['Proyecto'].astype(str) + "_" + \
                                          new_df['Año'].astype(str) + "_" + \
                                          new_df['Mes'].astype(str) + "_" + \
                                          new_df['Equipo_Comercial'].astype(str)
                    
                    # Find new records and updates
                    existing_keys = set(existing_df['unique_key'])
                    new_keys = set(new_df['unique_key'])
                    
                    # Records to add (completely new)
                    records_to_add = new_df[~new_df['unique_key'].isin(existing_keys)]
                    
                    # Records to update (existing but potentially different data)
                    records_to_update = new_df[new_df['unique_key'].isin(existing_keys)]
                    
                    # Remove old versions of records being updated
                    updated_existing_df = existing_df[~existing_df['unique_key'].isin(records_to_update['unique_key'])]
                    
                    # Combine: existing (minus updated) + new + updated
                    final_df = pd.concat([updated_existing_df, records_to_add, records_to_update], ignore_index=True)
                    
                    # Remove the temporary unique_key column
                    final_df = final_df.drop('unique_key', axis=1)
                    
                    print(f"Added {len(records_to_add)} new records")
                    print(f"Updated {len(records_to_update)} existing records")
                    
                except Exception as e:
                    print(f"Error reading existing file: {e}")
                    print("Creating new master file...")
                    final_df = new_df
            else:
                print("No existing master file - creating new one...")
                final_df = new_df
            
            # Sort the final data
            final_df = final_df.sort_values(['Proyecto', 'Año', 'Mes', 'Equipo_Comercial'])
            
            # Save to Excel
            final_df.to_excel(self.master_file_path, index=False, sheet_name='Extracted_Data')
            
            # Create clean summary table
            print(f"\nEXTRACTION SUMMARY")
            print("=" * 60)
            print(f"{'Total Records in Master:':<25} {len(final_df):>10}")
            print(f"{'New Records Added:':<25} {len(records):>10}")
            print(f"{'Master File Saved To:':<25} {str(self.master_file_path.absolute())}")
            print(f"{'Proyectos in Master:':<25} {final_df['Proyecto'].nunique():>10}")
            print(f"{'Years Found:':<25} {', '.join(sorted(final_df['Año'].dropna().astype(str).unique()))}")
            print(f"{'Months Found:':<25} {', '.join(sorted(final_df['Mes'].dropna().unique()))}")
            
            # Data breakdown table
            print("\nDATA BREAKDOWN BY PROYECTO:")
            print("-" * 60)
            print(f"{'Proyecto':<15} {'Records':<10} {'Years':<15} {'Months':<20}")
            print("-" * 60)
            
            for proyecto in sorted(final_df['Proyecto'].unique()):
                proyecto_data = final_df[final_df['Proyecto'] == proyecto]
                years = sorted(proyecto_data['Año'].dropna().astype(str).unique())
                months = sorted(proyecto_data['Mes'].dropna().unique())
                
                print(f"{proyecto:<15} {len(proyecto_data):<10} {', '.join(years):<15} {', '.join(months[:3])}{('...' if len(months) > 3 else ''):<20}")
            
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def run_fast_extraction(self, proyecto_name=None):
        """Main fast extraction method"""
        start_time = datetime.now()
        
        print("Excel Data Extractor - Starting...")
        print(f"Base Path: {self.base_path}")
        print(f"Target Sheet: {self.config['TARGET_SHEET_NAME']}")
        
        if proyecto_name:
            print(f"Processing Proyecto: {proyecto_name}")
        else:
            print("Processing: ALL proyectos")
        
        # Find all files
        print("\nScanning directories...")
        files_info = self.find_all_resumen_files(proyecto_name)
        
        if not files_info:
            print("ERROR: No RESUMEN files found!")
            return
        
        print(f"Found {len(files_info)} RESUMEN files")
        
        # Process files in parallel
        print("Processing files...")
        records = self.process_files_parallel(files_info)
        
        # Save results
        self.save_results(records)
        
        # Show timing
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\nPERFORMANCE METRICS:")
        print("-" * 30)
        print(f"{'Total Processing Time:':<20} {duration.total_seconds():.2f} seconds")
        print(f"{'Average Per File:':<20} {duration.total_seconds()/len(files_info):.2f} seconds")
        print(f"{'Files Processed:':<20} {len(files_info)}")
        print(f"{'Success Rate:':<20} {len(records)}/{len(files_info)} ({(len(records)/len(files_info)*100):.1f}%)")

def main():
    """Main function for fast extractor"""
    print("EXCEL DATA EXTRACTOR")
    print("=" * 50)
    print(f"Base Path: {CONFIG['BASE_PATH']}")
    print(f"Master File: {CONFIG['MASTER_FILE_PATH']}")
    print(f"Target Sheet: {CONFIG['TARGET_SHEET_NAME']}")
    print()
    
    # Always ask for proyecto input
    proyecto = input("Enter proyecto name (leave empty to process ALL proyectos): ").strip() or None
    
    if proyecto:
        print(f"Will process proyecto: {proyecto}")
    else:
        print("Will process: ALL proyectos")
    
    # Run extraction
    extractor = FastExcelExtractor()
    extractor.run_fast_extraction(proyecto_name=proyecto)

if __name__ == "__main__":
    main()
