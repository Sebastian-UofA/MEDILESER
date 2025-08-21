## VALORIZACION — Documentación (Español)

Versión: 1.0
Generado: 2025-08-18

Resumen
-------
Documento que explica cómo ejecutar, validar y mantener `VALORIZACION-CON-PLANTILLA.py`. Incluye entradas, procesos, el algoritmo de selección heredado, reglas de clasificación, consejos de rendimiento y resolución de problemas.

Contenido
--------
- Propósito
- Inicio rápido
- Archivos de entrada y columnas necesarias
- Heurísticas de detección de columnas
- Flujo de procesamiento
- Algoritmos principales
- Política DIARIO / WALKBY
- Memoria y rendimiento
- Resolución de problemas
- Validación y pruebas recomendadas
- Checklist de mantenimiento
- Apéndice: salidas y regeneración

Propósito
-------
`VALORIZACION` transforma exportes de lecturas de medidores en un libro Excel auditable. Salidas esperadas:

- Una tabla pivote (`medidor × fecha`) con recuento diario
- Clasificación por medidor (DIARIO, INTERMITENTE, WALKBY, etc.)
- `VALORIZABLE`: filas finales seleccionadas para valorización
- `ELIMINADOS`: lecturas removidas con motivo

Inicio rápido
-----------
1. Copia los archivos de entrada en una carpeta.
2. Ejecuta:

```powershell
py -3 VALORIZACION.py
```

O arrastra los archivos sobre el script en el orden: `data_file`, `meters_file` (opcional), `incidents_file` (opcional).

Se generará `dep_<input_basename>.xlsx` junto al archivo de datos.

Archivos de entrada
-----------
Archivo de datos (obligatorio)
- Excel `.xls`/`.xlsx`, puede contener varias hojas.
- Debe incluir: ID medidor, fecha o datetime (`Record Time`), columna de hora opcional y al menos una columna numérica.

Archivo de medidores (opcional)
- Archivo Excel con lista de medidores (una columna válida). Si se proporciona, medidores ausentes se marcarán `SIN LECTURA`.

Archivo de incidencias (opcional)
- Dos columnas: ID medidor y tipo de incidencia.

Heurísticas de columnas
-----------------------
Coincidencia por subcadena (case-insensitive):

- ID medidor: `meter id`, `meter_id`, `meterid`, `medidor`, `id_medidor`
- Fecha/DateTime: `record time` (evitar `time_time`)
- Hora: `time_time`, `time`
- Flujo/volumen: nombres con `flow`, `count`, `total`, `volume`.

Si existe una columna DateTime combinada, `split_datetime_column()` intentará dividirla.

Flujo de procesamiento
----------------------
1. Leer y concatenar todas las hojas.
2. Leer archivos opcionales (medidores/incidencias).
3. Optimizar memoria (downcast/`category`).
4. Separar filas por `Gateway` vs `WALKBY`.
5. Gateway: limitar lecturas por día a `MAX_READINGS_PER_DAY` (7) usando `remove_closest_readings_fast()` (paridad con legado).
6. Walkby: conservar la lectura más reciente por día.
7. Walkby puro: conservar la primera lectura del mes.
8. Registrar todas las eliminaciones (`Motivo_Eliminacion`).
9. Generar pivote y calcular `Numero De Lecturas` y `Dias Lecturadas`.
10. Aplicar clasificación mensual y resoluciones por incidencias.
11. Exportar libro Excel.

Algoritmos principales
----------------------
- `remove_closest_readings_fast()`
  - Portada del script legado para preservar selección idéntica: mantiene extremos y elige intermedios para maximizar dispersión temporal.
- WALKBY
  - Por día: conservar la más reciente.
  - Puro-mensual: conservar la primera lectura del mes.

Política DIARIO / WALKBY
-----------------------
1. Identificar medidores `DIARIO / WALKBY`.
2. Si una fecha tiene ambos tipos (Gateway + Walkby) eliminar la Walkby de ese día y marcar `DIARIO`.
3. Recalcular totales.
4. Si queda `DIARIO / WALKBY`, setear `INTERMITENTE / WALKBY`.

Memoria y rendimiento
---------------------
- `optimize_memory_usage()` reduce tipos para ahorrar RAM.
- La reducción por grupo puede correr en paralelo; para entradas muy grandes procesar por mes.
- Si aparece `ArrayMemoryError` aumenta RAM o divide datos.

Resolución de problemas
-----------------------
- Columnas faltantes: revisa `df.columns` y usa `split_datetime_column()` si procede.
- Errores de memoria: procesar recortes temporales o usar máquina con más memoria.
- Clasificación errónea: asegurarse de la versión más reciente del script que recalcule totales.

Validación y pruebas
--------------------
- Pruebas unitarias recomendadas:
  - Comparar la función `remove_closest_readings_fast()` contra el legado en datasets pequeños.
  - Probar `limit_readings_per_day_optimized()`.
  - Verificar `ELIMINADOS` y pivot con datos mixtos.

Checklist de mantenimiento
-------------------------
- Conservar `temp-valorization.py` como referencia canónica.
- No cambiar `remove_closest_readings_fast()` sin tests y documentación.
- Añadir tests (`tests/`) y configurar CI con `pytest`.

Apéndice: salidas y regeneración
--------------------------------
- Salida típica: `dep_<input>.xlsx` con hojas `TABLA DINAMICA`, `VALORIZABLE`, `ELIMINADOS`.

Regenerar PDF desde Markdown

Si actualizas cualquiera de los Markdown, genera los PDFs con:

```powershell
py -3 docs\\make_pdf.py docs\\VALORIZACION_DOCUMENTACION.md docs\\VALORIZACION_DOCUMENTACION.pdf
py -3 docs\\make_pdf.py docs\\VALORIZACION_DOCUMENTACION_ES.md docs\\VALORIZACION_DOCUMENTACION_ES.pdf
```

Entrega y traspaso
------------------
Deja estos recursos al equipo:

- La documentación EN/ES en `docs/`.
- Una demo corta ejecutando el script con ejemplo.
- El `temp-valorization.py` para comprobación de paridad.

Fin del documento
  - Testear `limit_readings_per_day_optimized()` con grupos de distintos tamaños.
