# Documentación para Scripts de Análisis de Medidores

Este documento proporciona instrucciones sobre cómo utilizar los scripts `RUTA.py` e `informe.py` para analizar datos de medidores de agua.

## Tabla de Contenidos
1.  [Descripción General](#descripción-general)
2.  [Configuración e Instalación](#configuración-e-instalación)
3.  [Flujo de Trabajo](#flujo-de-trabajo)
4.  [Script 1: `RUTA.py`](#script-1-rutapy)
    *   [Propósito](#propósito)
    *   [Archivos de Entrada](#archivos-de-entrada)
    *   [Archivo de Salida](#archivo-de-salida)
    *   [Cómo Ejecutar](#cómo-ejecutar)
5.  [Script 2: `informe.py`](#script-2-informepy)
    *   [Propósito](#propósito-1)
    *   [Archivos de Entrada](#archivos-de-entrada-1)
    *   [Archivo de Salida](#archivo-de-salida-1)
    *   [Cómo Ejecutar](#cómo-ejecutar-1)
6.  [Solución de Problemas](#solución-de-problemas)

---

## Descripción General

Este es un proceso de dos pasos que involucra dos scripts de Python:

1.  **`RUTA.py`**: Este script realiza el análisis inicial. Toma lecturas brutas de medidores, una lista de IDs de medidores y archivos opcionales de incidentes/alarmas. Luego, categoriza cada medidor basándose en un conjunto de reglas y produce un archivo de resumen llamado `Valorization_Summary.xlsx`.
2.  **`informe.py`**: Este script toma el archivo de lecturas brutas y el archivo `Valorization_Summary.xlsx` generado por `RUTA.py` para crear un informe final más detallado, `Informe_Ciclos.xlsx`. Este informe incluye valores de flujo máximo/mínimo y una lectura sugerida para los medidores que requieren atención.

**Es crucial ejecutar `RUTA.py` antes de ejecutar `informe.py`**, ya que el segundo script depende del resultado del primero.

---

## Configuración e Instalación

Antes de ejecutar los scripts, necesita tener Python instalado en su sistema y las librerías requeridas.

1.  **Instalar Python:** Si no tiene Python, descárguelo e instálelo desde el sitio web oficial [python.org](https://www.python.org/downloads/). Asegúrese de marcar la casilla que dice "Add Python to PATH" durante la instalación.

2.  **Instalar Librerías:** Abra una terminal o símbolo del sistema y ejecute el siguiente comando para instalar las librerías necesarias:

    ```bash
    pip install pandas openpyxl xlrd xlsxwriter
    ```

---

## Flujo de Trabajo

El flujo de trabajo estándar es el siguiente:

1.  Reúna todos sus archivos de entrada (IDs de Medidores, Lecturas, Incidentes, Alarmas).
2.  Ejecute `RUTA.py`.
3.  Use el resultado de `RUTA.py` (`Valorization_Summary.xlsx`) como entrada para `informe.py`.
4.  Ejecute `informe.py`.
5.  El informe final es `Informe_Ciclos.xlsx`.

---

## Script 1: `RUTA.py`

### Propósito

Analizar los datos de los medidores y clasificar cada uno en una de las siguientes categorías para tomar acción:
*   **RUTA RECOMENDADA**: Indica que un medidor necesita ser revisado en una ruta.
*   **INSPECCION**: Se necesita una revisión más urgente debido a alarmas específicas (Aire en Tubería, Flujo Inverso).
*   **AVISO**: Un aviso o advertencia para otros tipos de alarmas.
*   **Sin Recomendación**: No se detectaron problemas.

### Archivos de Entrada

El script abrirá diálogos de archivo pidiéndole que seleccione los siguientes archivos:

1.  **Archivo de Texto de IDs de Medidores** (`.txt`): Un archivo de texto plano que contiene los IDs de los medidores a analizar. El script busca específicamente IDs que comiencen con `KA` o `KB` seguidos de exactamente 8 números (ej., `KA12345678`).
2.  **Archivo Excel de Lecturas de Medidores** (`.xls` o `.xlsx`): Un archivo de Excel que contiene los datos brutos de lectura de los medidores. Debe contener al menos las columnas `Meter ID` y `Record Time`.
3.  **Archivo de Incidentes (Opcional)** (`.xls` o `.xlsx`): Un archivo de Excel que lista los medidores con incidentes conocidos.
4.  **Archivo de Alarmas (Opcional)** (`.xls` o `.xlsx`): Un archivo de Excel que lista el conteo de diferentes alarmas para cada medidor.

### Archivo de Salida

*   **`Valorization_Summary.xlsx`**: Un archivo de Excel que contiene dos hojas:
    1.  Una copia de todos los datos de lecturas brutas.
    2.  Una tabla dinámica que resume el análisis, con el estado de cada medidor resaltado. Este archivo se crea en el mismo directorio que el archivo de lecturas de entrada.

### Cómo Ejecutar

1.  Abra una terminal o símbolo del sistema.
2.  Navegue al directorio donde se encuentra el script.
3.  Ejecute el script usando el comando: `python RUTA.py`
4.  Aparecerán ventanas de diálogo de archivo. Seleccione los archivos requeridos en el orden en que se solicitan.

---

## Script 2: `informe.py`

### Propósito

Generar un informe final y detallado que incluye las lecturas de flujo máximo y mínimo para cada día y proporciona una lectura sugerida para los medidores que no tuvieron lectura en el penúltimo día.

### Archivos de Entrada

1.  **Archivo Excel de Lecturas de Medidores**: El **mismo** archivo de lecturas que usó para `RUTA.py`.
2.  **Archivo de Resumen de Valorización**: El archivo `Valorization_Summary.xlsx` que fue generado por `RUTA.py`.
3.  **Archivo de Incidentes (Opcional)**: Una versión actualizada del archivo de incidentes.
4.  **Archivo de Alarmas (Opcional)**: Una versión actualizada del archivo de alarmas.

### Archivo de Salida

*   **`Informe_Ciclos.xlsx`**: El informe final. Contiene tres hojas:
    1.  `Total Readings`: Una copia de los datos brutos.
    2.  `Meter Reading Pivot Table`: La tabla dinámica de resumen de `Valorization_Summary.xlsx`.
    3.  `Max-Min Flow Pivot Table`: La hoja principal del informe, que muestra los flujos diarios máximos/mínimos, el estado y las lecturas sugeridas.

### Cómo Ejecutar

1.  Asegúrese de haber ejecutado ya `RUTA.py` y de tener el archivo `Valorization_Summary.xlsx`.
2.  Abra una terminal o símbolo del sistema.
3.  Navegue al directorio donde se encuentra el script.
4.  Ejecute el script usando el comando: `python informe.py`
5.  Seleccione los archivos requeridos cuando se le solicite.

---

## Solución de Problemas

Aquí hay algunos problemas comunes y cómo resolverlos.

*   **Problema:** `ModuleNotFoundError: No module named 'pandas'` (u otra librería).
    *   **Solución:** No ha instalado las librerías requeridas. Ejecute el comando de instalación de la sección [Configuración e Instalación](#configuración-e-instalación).

*   **Problema:** El script se cierra con un error como `Error reading Excel file` o `Could not read or process incident file`.
    *   **Solución:**
        *   Asegúrese de que el archivo de Excel no esté abierto en otro programa mientras el script se está ejecutando.
        *   Verifique que el archivo no esté corrupto intentando abrirlo manualmente en Excel.
        *   El script espera ciertos nombres de columna (ej., `Record Time`, `Meter ID`, `TotalFlow`). Asegúrese de que sus archivos de entrada tengan los encabezados correctos.

*   **Problema:** El archivo de salida está vacío o le faltan muchos medidores.
    *   **Solución:**
        *   Verifique que los IDs de los medidores en su archivo de texto coincidan con el formato en el archivo de lecturas. El script añade el prefijo `00` a los IDs de medidor que comienzan con `KA` o `KB` para normalizarlos.
        *   Asegúrese de que el formato del ID del medidor en el archivo de texto sea correcto (`KA` o `KB` seguido de 8 dígitos). El script ignorará los IDs que no coincidan con este patrón.

*   **Problema:** La ventana de diálogo de archivo no aparece.
    *   **Solución:** La librería `tkinter`, que crea los diálogos, puede tener problemas en algunos sistemas. Asegúrese de que su instalación de Python sea estándar. La ventana también podría estar apareciendo detrás de otras ventanas abiertas.
