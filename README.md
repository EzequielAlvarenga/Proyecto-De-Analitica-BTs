# Proyecto BTS de retrasos de llegada

## Inicio rápido
1. Instalar Python 3.12 (entorno comprobado: 3.12.14).
2. Descomprimir el paquete y abrir una terminal dentro de proyecto_bts.
3. Crear entorno: `python -m venv .venv`.
4. Activar en Windows: `.venv\Scripts\activate`; en macOS/Linux: `source .venv/bin/activate`.
5. Instalar: `python -m pip install -r requirements.txt`.
6. Ejecutar: `python src/pipeline.py`.
7. Tablero: `streamlit run dashboard.py`. Se abrirá en el navegador local.

Sin programar ni iniciar servidor: abrir reports/dashboard_interactivo.html. El HTML contiene Plotly y funciona sin internet; ofrece un selector por origen. Streamlit permite filtros combinados y descarga.

## Entregables
- `src/pipeline.py`: pipeline comentado, fuente única del análisis.
- `notebooks/proyecto.ipynb`: guía ejecutable del pipeline.
- `data/raw`: ZIP oficial íntegro y diccionario del proveedor.
- `data/processed/vuelos.csv`: datos procesados en unidades indicadas.
- `models/arbol.pkl` y `reglas.txt`: preprocesamiento y árbol; no cargar modelos de origen desconocido.
- `reports/Reporte_tecnico.docx`: reporte de 8 páginas previsto.
- 
- `reports/resultados.json`: auditoría, hash, contraste y métricas.
- `reports/figures`: evidencia gráfica; CSV adicionales con descriptivos y correlaciones.

## Ejecución del notebook
Abrir con Jupyter, VS Code o Google Colab y seleccionar un entorno con requirements.txt instalado. En Colab subir y descomprimir primero el ZIP del proyecto, entrar a proyecto_bts y ejecutar las celdas. El notebook llama al mismo script: no mantiene una segunda implementación.

## Alcance y autoría
Vuelos domésticos estadounidenses de enero de 2024. Excluye cancelados y desviados de la etiqueta de llegada; los contabiliza en auditoría. No generalizar a todo el año ni a El Salvador. Modelo orientativo no certificado. Implementación original con asistencia de IA: declarar según las reglas del docente. El equipo debe ejecutar y entender la entrega.

## Reproducción de las salidas
Las gráficas, tabla procesada, modelo y resultados JSON se reconstruyen al ejecutar el pipeline. Los documentos son entregables editoriales que describen esta ejecución fija; si cambia el mes o el código, actualizar sus cifras antes de entregar. La semilla es 42. El script descarga desde BTS si falta el ZIP, por lo que esa situación requiere internet.
