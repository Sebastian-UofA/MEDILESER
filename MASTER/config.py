"""
Configuration file for Excel Data Extractor
Modify these settings according to your needs
"""

# Base configuration
CONFIG = {
    # Main paths
    'BASE_PATH': "Z:\\7. FACTURACIÓN",
    'MASTER_FILE_PATH': "master.xlsx",  # Will be saved in current directory
    
    # Specific proyecto to process (None = process all, will ask user)
    'SPECIFIC_PROYECTO': None,  # Set to None to always ask user for proyecto

    # Excel sheet name to look for
    'TARGET_SHEET_NAME': "DATOS",
    
    # Data fields to extract (these are the row identifiers in the DATOS sheet)
    'DATA_FIELDS': {
        'lecturas': ['LECTURAS', '#'],  # Must contain both words
        'cantidad_medidores': ['CANTIDAD', 'MEDIDORES'],
        'valorizable_gateway': ['VALORIZABLE', 'GATEWAY'],
        'valorizable_walkby': ['VALORIZABLE', 'WALKBY']
    },
    
    # File patterns
    'RESUMEN_FILE_PREFIX': 'RESUMEN',
    'EXCEL_EXTENSIONS': ['.xlsx', '.xls'],
    
    # Folder patterns
    'YEAR_FOLDER_PATTERN': r'AÑO\s+\d{4}',  # Matches "AÑO 2025"
    
    # Logging level
    'LOG_LEVEL': 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR
}

# Spanish months mapping
MONTHS_SPANISH = {
    'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4,
    'MAYO': 5, 'JUNIO': 6, 'JULIO': 7, 'AGOSTO': 8,
    'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
}
