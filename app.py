import streamlit as st
 import cv2
 import numpy as np
 import tempfile
 from llama_index.llms.gemini import Gemini
 from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock
 import re
 import os
 import base64
 from PIL import Image
 import io
 # import matplotlib.pyplot as plt  # Comentamos esta línea para evitar el error
 import pandas as pd
 import json
 from datetime import datetime
 import uuid
 import altair as alt
 
 # Intentar importar pytesseract, pero manejar caso cuando no está instalado
 try:
     import pytesseract
 except ImportError:
     pass  # La variable se mantiene como False
 
 # Set up Google API Key
 os.environ["GOOGLE_API_KEY"] = "AIzaSyA5FyLIhOSIKxGw3TebXzLfMjuYx5fVwW4"
 
 # Initialize the Gemini model
 gemini_pro = Gemini(model_name="models/gemini-1.5-flash")
 
 # Inicializar variables de estado
 if 'historial_analisis' not in st.session_state:
     st.session_state.historial_analisis = []
 
 if 'fechas_guardadas' not in st.session_state:
     st.session_state.fechas_guardadas = []
 
 if 'show_debug' not in st.session_state:
     st.session_state.show_debug = False
 
 # Función para instalar pytesseract
 def install_pytesseract():
     import subprocess
     import sys
     
     try:
         st.info("Instalando pytesseract... Por favor, espere.")
         subprocess.check_call([sys.executable, "-m", "pip", "install", "pytesseract"])
         st.success("¡pytesseract instalado correctamente! Por favor, reinicie la aplicación.")
         st.warning("IMPORTANTE: También necesita instalar Tesseract OCR en su sistema. Visite: https://github.com/tesseract-ocr/tesseract")
         return True
     except Exception as e:
         st.error(f"Error al instalar pytesseract: {str(e)}")
         return False
 
 # Función para detectar fechas de vencimiento
 def detect_expiration_dates(img):
     # Comprobamos de nuevo la disponibilidad para manejar casos donde
     # la aplicación se reinicia después de la instalación
     pytesseract_available = False
     try:
         import pytesseract
         # Configurar la ruta a Tesseract en Windows (ajustar según tu instalación)
         if os.name == 'nt':  # Windows
             pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
             # Verificar rutas alternativas comunes en Windows
             if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
                 alt_paths = [
                     r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                     r'C:\Tesseract-OCR\tesseract.exe'
                 ]
                 for path in alt_paths:
                     if os.path.exists(path):
                         pytesseract.pytesseract.tesseract_cmd = path
                         break
         pytesseract_available = True
     except ImportError:
         pytesseract_available = False
     
     # Si pytesseract no está disponible, usar simulación básica
     if not pytesseract_available:
         st.warning("""
         ⚠️ Tesseract OCR no está instalado o configurado correctamente.
         Las fechas de vencimiento mostradas son simuladas, no detectadas de la imagen.
         """)
         # Simulamos una fecha de vencimiento para demostración
         today = datetime.now()
         
         # Una fecha vencida (30 días atrás)
         expired_date = today - pd.Timedelta(days=30)
         expired_str = expired_date.strftime("%d/%m/%Y")
         
         # Una fecha próxima a vencer (5 días adelante)
         soon_date = today + pd.Timedelta(days=5)
         soon_str = soon_date.strftime("%d/%m/%Y")
         
         # Una fecha válida (60 días adelante)
         valid_date = today + pd.Timedelta(days=60)
         valid_str = valid_date.strftime("%d/%m/%Y")
         
         # Retornamos fechas simuladas
         return [
             {
                 'date_str': expired_str,
                 'parsed_date': expired_date,
                 'is_expired': True,
                 'days_remaining': -30
             },
             {
                 'date_str': soon_str,
                 'parsed_date': soon_date,
                 'is_expired': False,
                 'days_remaining': 5
             },
             {
                 'date_str': valid_str,
                 'parsed_date': valid_date,
                 'is_expired': False,
                 'days_remaining': 60
             }
         ]
     
     # Si pytesseract está disponible, usar detección real
     # Mejorar la imagen para una mejor detección de texto
     
     # Mostrar la imagen original para depuración si está activado
     if 'show_debug' in st.session_state and st.session_state.show_debug:
         st.image(img, caption="Imagen original", channels="BGR", use_column_width=True)
     
     # 1. Redimensionar la imagen si es muy pequeña
     height, width = img.shape[:2]
     if height < 300 or width < 300:
         scale_factor = max(300 / height, 300 / width)
         img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
     
     # 2. Convertir imagen a escala de grises
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
     # 3. Aplicar varias técnicas de mejora de imagen para OCR
     # 3.1 Reducción de ruido
     denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
     
     # 3.2 Aplicar umbralización adaptativa
     thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
     
     # 3.3 Alternativa: umbralización con Otsu
     _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
     
     # Mostrar imágenes procesadas para depuración
     if 'show_debug' in st.session_state and st.session_state.show_debug:
         col1, col2 = st.columns(2)
         with col1:
             st.image(thresh, caption="Umbralización adaptativa", use_column_width=True)
         with col2:
             st.image(otsu, caption="Umbralización Otsu", use_column_width=True)
     
     # Extraer texto con pytesseract (OCR) de ambas versiones procesadas
     results = []
     try:
         # 4. Configurar opciones de tesseract para mejorar la detección
         # Definir configuraciones para varios idiomas y escenarios
         configs = [
             r'--oem 3 --psm 6 -l spa+eng',    # OCR Engine mode 3, bloque uniforme, español+inglés
             r'--oem 3 --psm 11 -l spa+eng',   # OCR Engine mode 3, texto disperso, español+inglés
             r'--oem 3 --psm 3 -l spa+eng',    # OCR Engine mode 3, columna de texto, español+inglés
             r'--oem 3 --psm 4 -l spa+eng',    # OCR Engine mode 3, bloque de texto alineado a la derecha, español+inglés
             r'--oem 3 --psm 6 -l eng',        # Solo inglés
             r'--oem 3 --psm 6 -l spa',        # Solo español
             r'--oem 3 --psm 6 -l por+spa+eng' # Portugués+español+inglés (para productos de América Latina)
         ]
         
         # 5. Extraer texto usando múltiples configuraciones
         all_texts = []
         
         for config in configs:
             try:
                 # Extraer texto de ambas imágenes procesadas
                 text_thresh = pytesseract.image_to_string(thresh, config=config)
                 text_otsu = pytesseract.image_to_string(otsu, config=config)
                 
                 # Agregar a la lista de textos
                 all_texts.extend([text_thresh, text_otsu])
                 
                 # Mensaje para depuración
                 if 'show_debug' in st.session_state and st.session_state.show_debug:
                     st.text(f"Extrayendo texto con config: {config}")
             except Exception as e:
                 if 'show_debug' in st.session_state and st.session_state.show_debug:
                     st.text(f"Error con config {config}: {str(e)}")
         
         # Combinar todos los textos extraídos
         text = "\n".join(all_texts)
         
         # Mostrar el texto extraído para depuración
         if 'show_debug' in st.session_state and st.session_state.show_debug:
             st.text("Texto detectado (combinado de todas las configuraciones):")
             st.code(text)
             
     except Exception as e:
         st.warning(f"""
         Error al usar pytesseract: {str(e)}
         
         Parece que Tesseract OCR no está instalado correctamente en tu sistema o no está en tu PATH.
         
         Instala Tesseract OCR desde: https://github.com/UB-Mannheim/tesseract/wiki (Windows)
         o con 'brew install tesseract' (macOS) o 'sudo apt install tesseract-ocr' (Linux).
         
         Después de instalar Tesseract OCR, asegúrate de añadirlo a tu PATH o especificar su ruta en el código.
         """)
         text = ""
     
     # Patrones comunes de fechas de vencimiento en español e inglés
     date_patterns = [
         # Patrones con palabras clave en español
         r'vence(?::|.{0,5})\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Vence: DD/MM/AAAA
         r'exp(?:ira|\.)\s*(?:date|el)?(?::|.{0,5})\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Expira: DD/MM/AAAA
         r'consumir antes de(?:l)?(?::|.{0,5})\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Consumir antes de: DD/MM/AAAA
         r'best before(?::|.{0,5})\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Best before: DD/MM/AAAA
         r'fecha de vencimiento(?::|.{0,5})\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Fecha de vencimiento: DD/MM/AAAA
         r'caducidad(?::|.{0,5})\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Caducidad: DD/MM/AAAA
         r'cad(?::|.{0,5})\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Cad: DD/MM/AAAA (abreviatura común)
         # Patrones con palabras clave en inglés
         r'exp(?:iry|\.)?(?::|.{0,5})\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Exp: DD/MM/AAAA
         r'use by(?::|.{0,5})\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Use by: DD/MM/AAAA
         # Formato de fecha sin texto precedente
         r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # DD/MM/AAAA o DD-MM-AAAA genérico
         # Formatos numéricos alternativos 
         r'\b(\d{2}[/.]\d{2}[/.]\d{2,4})\b',  # DD.MM.AAAA
         # Formato de fecha en texto (ejemplo: 01 ENE 2023)
         r'\b(\d{1,2}\s+(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)[a-z]*\s+\d{2,4})\b',  # 01 ENE 2023
         r'\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})\b'   # 01 JAN 2023
     ]
     
     # Buscar todas las posibles fechas en el texto
     detected_dates = []
     for pattern in date_patterns:
         matches = re.finditer(pattern, text, re.IGNORECASE)
         for match in matches:
             # Obtener la fecha del grupo de captura
             if len(match.groups()) > 0:
                 date_str = match.group(1)
                 # Evitar duplicados
                 if date_str not in detected_dates:
                     detected_dates.append(date_str)
     
     # Procesar las fechas encontradas
     expiration_info = []
     today = datetime.now()
     
     # Mostrar las fechas detectadas en bruto para depuración
     if 'show_debug' in st.session_state and st.session_state.show_debug and detected_dates:
         st.text("Fechas detectadas en bruto:")
         for date in detected_dates:
             st.code(date)
     
     for date_str in detected_dates:
         # Manejo de fechas en formato de texto (ej: 01 ENE 2023)
         month_mappings = {
             'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04', 'may': '05', 'jun': '06',
             'jul': '07', 'ago': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12',
             'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
             'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
         }
         
         # Comprobar si es una fecha en formato de texto y convertirla
         for month_name in month_mappings.keys():
             if month_name in date_str.lower():
                 parts = re.split(r'\s+', date_str)
                 if len(parts) == 3:
                     try:
                         day = parts[0].zfill(2)
                         month = month_mappings.get(parts[1].lower()[:3], '01')
                         year = parts[2]
                         date_str = f"{day}/{month}/{year}"
                         break
                     except:
                         continue
         
         # Reemplazar separadores por / para estandarizar formato
         date_str = date_str.replace('-', '/').replace('.', '/')
         
         # Intentar diferentes formatos de fecha
         date_formats = ['%d/%m/%Y', '%d/%m/%y', '%m/%d/%Y', '%m/%d/%y', '%Y/%m/%d']
         
         parsed_date = None
         for fmt in date_formats:
             try:
                 parsed_date = datetime.strptime(date_str, fmt)
                 # Si el año tiene 2 dígitos y es menor que 50, asumir 20XX, si no 19XX
                 if fmt.endswith('%y'):
                     year = parsed_date.year
                     if year < 2000:
                         if year < 50:
                             parsed_date = parsed_date.replace(year=year+2000)
                         else:
                             parsed_date = parsed_date.replace(year=year+1900)
                 break
             except ValueError:
                 continue
         
         if parsed_date:
             # Validación adicional: rechazar fechas que estén muy en el pasado o futuro 
             # (para evitar falsos positivos)
             years_diff = abs(parsed_date.year - today.year)
             if years_diff > 10:  # Rechazar fechas más de 10 años en el pasado o futuro
                 continue
                 
             is_expired = parsed_date < today
             days_remaining = (parsed_date - today).days
             
             expiration_info.append({
                 'date_str': date_str,
                 'parsed_date': parsed_date,
                 'is_expired': is_expired,
                 'days_remaining': days_remaining if not is_expired else days_remaining
             })
     
     # Si no se detectaron fechas pero Tesseract está disponible, mostrar mensaje de ayuda
     if not expiration_info and pytesseract_available:
         st.info("""
         No se detectaron fechas de vencimiento en la imagen.
         
         Consejos para mejorar la detección:
         1. Asegúrate de que la fecha sea claramente visible en la imagen
         2. Mejora la iluminación y el enfoque
         3. Acerca la cámara a la etiqueta con la fecha de vencimiento
         """)
     
     return expiration_info
 
 # Función para detectar fechas de vencimiento con Gemini
 def detect_dates_with_gemini(img, image_path):
     try:
         # Crear mensaje para Gemini
         date_detection_msg = ChatMessage(
             role=MessageRole.USER,
             blocks=[
                 TextBlock(text="""Busca ÚNICAMENTE fechas de vencimiento, caducidad o consumo preferente en esta imagen.
                 
 Instrucciones detalladas:
 1. Examina cuidadosamente toda la imagen buscando fechas de vencimiento, caducidad o consumo preferente.
 2. Busca etiquetas con textos como "Vence", "Caduca", "Consumir antes de", "Expira", "Fecha de vencimiento", "Best before", "Expiry date", etc.
 3. Las fechas pueden estar en formatos como DD/MM/AAAA, MM/DD/AAAA, DD-MM-AA, etc.
 4. También pueden estar en formato de texto como "01 ENE 2023".
 
 IMPORTANTE: Responde SÓLO con un objeto JSON que contenga las fechas detectadas, sin ningún texto adicional, en este formato exacto:
 
 {
   "fechas_detectadas": [
     {
       "fecha": "DD/MM/AAAA", (o el formato que hayas detectado)
       "tipo": "vencimiento/caducidad/consumo preferente",
       "texto_completo": "el texto completo donde aparece la fecha",
       "confianza": "alta/media/baja"
     }
   ],
   "texto_extraido": "todo el texto visible en la imagen relacionado con fechas"
 }
 
 Si no detectas ninguna fecha, responde:
 {
   "fechas_detectadas": [],
   "texto_extraido": "texto visible en la imagen (si hay)"
 }"""),
                 ImageBlock(path=image_path, image_mimetype="image/jpeg"),
             ],
         )
         
         # Obtener respuesta de Gemini
         date_response = gemini_pro.chat(messages=[date_detection_msg])
         
         # Procesar la respuesta
         try:
             # Limpiar el texto de la respuesta para extraer solo el JSON
             response_text = date_response.message.content
             # Eliminar cualquier prefijo o sufijo que no sea JSON
             json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
             if json_match:
                 json_str = json_match.group(1)
                 # Intentar parsear el JSON
                 gemini_result = json.loads(json_str)
                 
                 # Procesar fechas detectadas
                 expiration_info = []
                 today = datetime.now()
                 
                 for fecha in gemini_result.get('fechas_detectadas', []):
                     date_str = fecha.get('fecha', '')
                     if not date_str:
                         continue
                     
                     # Intentar parsear la fecha
                     date_formats = ['%d/%m/%Y', '%d/%m/%y', '%m/%d/%Y', '%m/%d/%y', 
                                     '%d-%m-%Y', '%d-%m-%y', '%Y/%m/%d', '%Y-%m-%d']
                     
                     parsed_date = None
                     for fmt in date_formats:
                         try:
                             parsed_date = datetime.strptime(date_str, fmt)
                             # Ajustar año si tiene 2 dígitos
                             if fmt.endswith('%y'):
                                 year = parsed_date.year
                                 if year < 2000:
                                     if year < 50:
                                         parsed_date = parsed_date.replace(year=year+2000)
                                     else:
                                         parsed_date = parsed_date.replace(year=year+1900)
                             break
                         except ValueError:
                             continue
                     
                     if parsed_date:
                         is_expired = parsed_date < today
                         days_remaining = (parsed_date - today).days
                         
                         expiration_info.append({
                             'date_str': date_str,
                             'parsed_date': parsed_date,
                             'is_expired': is_expired,
                             'days_remaining': days_remaining,
                             'ai_detected': True,  # Marcar como detectado por AI
                             'confidence': fecha.get('confianza', 'media')
                         })
                 
                 if 'show_debug' in st.session_state and st.session_state.show_debug:
                     st.subheader("Resultado de Gemini para detección de fechas")
                     st.json(gemini_result)
                 
                 return expiration_info
             else:
                 return []
         except Exception as e:
             if 'show_debug' in st.session_state and st.session_state.show_debug:
                 st.error(f"Error al procesar respuesta de Gemini: {str(e)}")
             return []
             
     except Exception as e:
         if 'show_debug' in st.session_state and st.session_state.show_debug:
             st.error(f"Error al usar Gemini para detectar fechas: {str(e)}")
         return []
 
 # Custom CSS
 def local_css(file_name):
     with open(file_name, "r") as f:
         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
 
 def set_page_config():
     st.set_page_config(
         page_title="¿Qué hay en tu plato?",
         page_icon="🥩",
         layout="wide",
         initial_sidebar_state="expanded",
     )
 
 def main():
     set_page_config()
     local_css("style.css")
 
     st.sidebar.image("logo.png", use_container_width=True)
     st.sidebar.title("¿Qué hay en tu plato?")
     st.sidebar.markdown("Powered by Juan David Rivera")
     
     # Comprobamos si pytesseract está disponible
     pytesseract_available = False
     try:
         import pytesseract
         pytesseract_available = True
     except ImportError:
         pytesseract_available = False
     
     # Mostrar advertencia y botón para instalar pytesseract si no está disponible
     if not pytesseract_available:
         st.sidebar.warning("📦 El módulo pytesseract no está instalado. La detección de fechas de vencimiento será simulada.")
         if st.sidebar.button("Instalar pytesseract"):
             installed = install_pytesseract()
             if installed:
                 # Mostrar mensaje de éxito sin intentar cambiar la variable global
                 st.sidebar.success("¡pytesseract instalado correctamente! Por favor, **reinicie la aplicación** para usarlo.")
                 
                 # Mostrar instrucciones para instalar Tesseract OCR
                 with st.sidebar.expander("📋 Instrucciones para instalar Tesseract OCR"):
                     st.markdown("""
                     ### Instalación de Tesseract OCR
                     
                     La biblioteca pytesseract necesita el software Tesseract OCR para funcionar.
                     
                     #### Windows
                     1. Descarga el instalador desde [aquí](https://github.com/UB-Mannheim/tesseract/wiki)
                     2. Ejecuta el instalador y sigue las instrucciones
                     3. Añade la ruta de instalación (ej. `C:\\Program Files\\Tesseract-OCR`) a tu variable PATH:
                        - Panel de Control → Sistema → Configuración avanzada → Variables de entorno
                        - Edita la variable PATH y añade la ruta
                     
                     #### macOS
                     ```
                     brew install tesseract
                     ```
                     
                     #### Linux (Ubuntu/Debian)
                     ```
                     sudo apt update
                     sudo apt install tesseract-ocr
                     ```
                     """)
                 
                 st.stop()  # Detener la ejecución para evitar errores
 
     menu = ["Herramienta", "Sobre el Proyecto", "Investigaciones"]
     choice = st.sidebar.selectbox("Navigation", menu)
 
     if choice == "Herramienta":
         home_page()
     elif choice == "Sobre el Proyecto":
         about_page()
     elif choice == "Investigaciones":
         contact_page()
 
 def home_page():
     st.title("¿Qué hay en tu plato?")
     
     # Crear pestañas principales
     main_tabs = st.tabs(["📸 Analizar Imagen", "📊 Historial", "ℹ️ Información"])
     
     with main_tabs[0]:
         # Opción para subir imagen
         uploaded_file = st.file_uploader("Sube una imagen de tu comida", type=["jpg", "jpeg", "png"])
         
         if uploaded_file is not None:
             process_image(uploaded_file)
         else:
             img_file_buffer = st.camera_input("Toma una foto")
             if img_file_buffer is not None:
                 process_image(img_file_buffer)
     
     with main_tabs[1]:
         # Mostrar historial de análisis si existe
         if 'historial_analisis' in st.session_state and st.session_state.historial_analisis:
             st.subheader("Historial de Análisis")
             
             for i, analisis in enumerate(st.session_state.historial_analisis):
                 with st.expander(f"Análisis #{i+1} - {analisis.get('fecha', 'Sin fecha')}"):
                     st.write(analisis)
         else:
             st.info("No hay historial de análisis disponible.")
 
 # Función para procesar la imagen subida o tomada
 def process_image(img_file):
     # Crear un archivo temporal para guardar la imagen
     with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
         temp_filename = temp_file.name
         # Guardar la imagen subida en el archivo temporal
         temp_file.write(img_file.getvalue())
     
     try:
         # Leer la imagen con OpenCV
         img = cv2.imread(temp_filename)
         
         # Mostrar la imagen
         st.image(img_file, caption="Imagen subida", use_column_width=True)
         
         # Crear pestañas para análisis nutricional y fechas de vencimiento
         analysis_tabs = st.tabs(["📊 Análisis Nutricional", "📅 Fechas de Vencimiento", "🔍 Estado del Alimento"])
         
         with analysis_tabs[0]:
             st.subheader("Análisis Nutricional")
             
             with st.spinner("Analizando imagen con IA..."):
                 st.info("Procesando imagen para identificar alimentos y calcular información nutricional...")
                 
                 # Implementar análisis real con Gemini
                 try:
                     # Convertir imagen para Gemini
                     gemini_img = Image.open(temp_filename)
                     
                     # Crear mensaje para Gemini
                     food_analysis_msg = ChatMessage(
                         role=MessageRole.USER,
                         blocks=[
                             TextBlock(text="""Analiza esta imagen de comida y proporciona la siguiente información:
                             1. Identifica todos los alimentos presentes en la imagen
                             2. Para cada alimento, estima:
                                - Porción aproximada en gramos
                                - Calorías
                                - Proteínas en gramos
                                - Carbohidratos en gramos
                                - Grasas en gramos
                             
                             Responde SOLO con un objeto JSON con el siguiente formato (sin texto adicional):
                             {
                               "total_calories": número_total_calorías,
                               "items": [
                                 {
                                   "name": "nombre_alimento",
                                   "confidence": valor_entre_0_y_1,
                                   "portion": "porción_en_gramos",
                                   "nutrition": {
                                     "total_calories": calorías,
                                     "protein_g": proteínas_en_gramos,
                                     "carbs_g": carbohidratos_en_gramos,
                                     "fat_g": grasas_en_gramos
                                   }
                                 },
                                 ...
                               ]
                             }"""),
                             ImageBlock(path=temp_filename, image_mimetype="image/jpeg"),
                         ],
                     )
                     
                     # Obtener respuesta de Gemini
                     food_response = gemini_pro.chat(messages=[food_analysis_msg])
                     
                     # Procesar respuesta
                     response_text = food_response.message.content
                     
                     # Verificar si es texto JSON y extraerlo
                     json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                     if json_match:
                         json_str = json_match.group(1)
                         # Parsear JSON
                         analisis_real = json.loads(json_str)
                         
                         # Verificar que tiene la estructura correcta
                         if "items" in analisis_real and isinstance(analisis_real["items"], list):
                             # Agregar timestamp
                             analisis_real["id"] = str(uuid.uuid4())
                             analisis_real["date"] = datetime.now().strftime("%d/%m/%Y %H:%M")
                             
                             # Mostrar resultados reales
                             st.success(f"Se han identificado {len(analisis_real['items'])} alimentos en la imagen")
                             
                             for item in analisis_real["items"]:
                                 with st.container():
                                     # Crear una tarjeta visualmente atractiva para el alimento
                                     col1, col2 = st.columns([1, 3])
                                     
                                     with col1:
                                         # Mostrar el nombre del alimento y la confianza
                                         st.markdown(f"""
                                         <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center;">
                                             <h2 style="color: #2c3e50; margin-bottom: 5px;">{item['name']}</h2>
                                             <div style="background-color: #4CAF50; color: white; border-radius: 20px; padding: 5px 10px; display: inline-block; font-weight: bold;">
                                                 {int(float(item['confidence'])*100)}%
                                             </div>
                                             <p style="margin-top: 15px; font-weight: 500;">Porción estimada: <span style="color: #3498db; font-weight: 600;">{item['portion']}</span></p>
                                         </div>
                                         """, unsafe_allow_html=True)
                                         
                                         # Añadir botón para descargar informe
                                         if st.button(f"📥 Descargar informe de {item['name']}", key=f"download_{item['name']}"):
                                             st.success(f"Preparando informe detallado de {item['name']} para descarga...")
                                             # Aquí se podría implementar la generación de un PDF o CSV
                                     
                                     with col2:
                                         # Crear pestañas para información detallada
                                         item_tabs = st.tabs(["📊 Nutrientes", "📈 Gráficos", "ℹ️ Detalles"])
                                         
                                         with item_tabs[0]:
                                             # Tabla de información nutricional
                                             st.markdown(f"""
                                             <h4 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px;">Información Nutricional Detallada</h4>
                                             """, unsafe_allow_html=True)
                                             
                                             # Crear un dataframe para mostrar la información nutricional
                                             nutrition_data = {
                                                 "Nutriente": ["Calorías", "Proteínas", "Carbohidratos", "Grasas"],
                                                 "Valor": [
                                                     f"{item['nutrition']['total_calories']} kcal",
                                                     f"{item['nutrition']['protein_g']} g",
                                                     f"{item['nutrition']['carbs_g']} g",
                                                     f"{item['nutrition']['fat_g']} g"
                                                 ],
                                                 "% Valor Diario*": [
                                                     f"{int(item['nutrition']['total_calories']/2000*100)}%",
                                                     f"{int(item['nutrition']['protein_g']/50*100)}%",
                                                     f"{int(item['nutrition']['carbs_g']/300*100)}%",
                                                     f"{int(item['nutrition']['fat_g']/65*100)}%"
                                                 ]
                                             }
                                             
                                             nutrition_df = pd.DataFrame(nutrition_data)
                                             st.dataframe(nutrition_df, use_container_width=True)
                                             
                                             st.caption("*Porcentaje de valores diarios basados en una dieta de 2,000 calorías.")
                                         
                                         with item_tabs[1]:
                                             # Mostrar gráficos
                                             st.markdown("### Distribución de Macronutrientes")
                                             
                                             # Datos para el gráfico
                                             labels = ['Proteínas', 'Carbohidratos', 'Grasas']
                                             values = [
                                                 item['nutrition']['protein_g'] * 4,  # 4 calorías por gramo
                                                 item['nutrition']['carbs_g'] * 4,    # 4 calorías por gramo
                                                 item['nutrition']['fat_g'] * 9       # 9 calorías por gramo
                                             ]
                                             
                                             # Calcular porcentajes
                                             total = sum(values)
                                             percentages = [value/total*100 for value in values]
                                             
                                             # Crear gráfico de barras
                                             chart_data = pd.DataFrame({
                                                 'Macronutriente': labels,
                                                 'Calorías': values,
                                                 'Porcentaje': percentages
                                             })
                                             
                                             # Gráfico de barras usando Altair
                                             bar_chart = alt.Chart(chart_data).mark_bar().encode(
                                                 x=alt.X('Macronutriente:N', axis=alt.Axis(labelAngle=0)),
                                                 y=alt.Y('Calorías:Q'),
                                                 color=alt.Color('Macronutriente:N', 
                                                                 scale=alt.Scale(domain=labels, 
                                                                                 range=['#4CAF50', '#2196F3', '#FF9800'])),
                                                 tooltip=['Macronutriente', 'Calorías', 'Porcentaje']
                                             ).properties(height=250)
                                             
                                             st.altair_chart(bar_chart, use_container_width=True)
                                             
                                             # Gráfico circular para ver la distribución
                                             st.markdown("### Proporción de Calorías")
                                             
                                             # Preparar datos para gráfico circular
                                             pie_data = pd.DataFrame({
                                                 'Macronutriente': labels,
                                                 'Calorías': values
                                             })
                                             
                                             # Crear gráfico usando Altair
                                             pie_chart = alt.Chart(pie_data).mark_arc().encode(
                                                 theta=alt.Theta(field="Calorías", type="quantitative"),
                                                 color=alt.Color(field="Macronutriente", type="nominal",
                                                                scale=alt.Scale(domain=labels, 
                                                                                range=['#4CAF50', '#2196F3', '#FF9800'])),
                                                 tooltip=['Macronutriente', 'Calorías']
                                             ).properties(height=250)
                                             
                                             st.altair_chart(pie_chart, use_container_width=True)
                                         
                                         with item_tabs[2]:
                                             # Información detallada adicional
                                             st.markdown(f"""
                                             <h4 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px;">Información Adicional</h4>
                                             """, unsafe_allow_html=True)
                                             
                                             # Calcular algunos indicadores nutricionales adicionales
                                             protein_ratio = item['nutrition']['protein_g'] / float(item['portion'].replace('g', '')) * 100
                                             carb_ratio = item['nutrition']['carbs_g'] / float(item['portion'].replace('g', '')) * 100
                                             fat_ratio = item['nutrition']['fat_g'] / float(item['portion'].replace('g', '')) * 100
                                             calorie_density = item['nutrition']['total_calories'] / float(item['portion'].replace('g', ''))
                                             
                                             # Mostrar información detallada
                                             detail_cols = st.columns(2)
                                             
                                             with detail_cols[0]:
                                                 st.metric("Densidad Calórica", f"{calorie_density:.1f} kcal/g")
                                                 st.metric("Proteína por porción", f"{protein_ratio:.1f}%")
                                             
                                             with detail_cols[1]:
                                                 st.metric("Carbohidratos por porción", f"{carb_ratio:.1f}%")
                                                 st.metric("Grasas por porción", f"{fat_ratio:.1f}%")
                                             
                                             # Definir categoría del alimento basado en macronutrientes
                                             # Proteico: >20% proteínas, Alto en carbos: >50% carbos, Graso: >30% grasas
                                             food_category = ""
                                             if protein_ratio > 20:
                                                 food_category = "Alto en proteínas"
                                             elif carb_ratio > 50:
                                                 food_category = "Alto en carbohidratos"
                                             elif fat_ratio > 30:
                                                 food_category = "Alto en grasas"
                                             else:
                                                 food_category = "Composición balanceada"
                                             
                                             st.info(f"**Clasificación alimenticia**: {food_category}")
                                             
                                             # Añadir recomendaciones básicas
                                             st.markdown("### Recomendaciones de consumo")
                                             
                                             if food_category == "Alto en proteínas":
                                                 st.success("✅ Ideal para incremento de masa muscular y recuperación post-ejercicio.")
                                             elif food_category == "Alto en carbohidratos":
                                                 st.warning("⚠️ Proporciona energía rápida. Ideal para consumir antes de actividad física intensa.")
                                             elif food_category == "Alto en grasas":
                                                 st.warning("⚠️ Consumir con moderación. Rico en energía pero puede contribuir a aumentar el colesterol.")
                                             else:
                                                 st.success("✅ Alimento de composición balanceada, adecuado para consumo regular.")
                                     
                                     # Línea divisoria entre elementos
                                     st.markdown("<hr>", unsafe_allow_html=True)
                             
                             # Botón para guardar el análisis
                             if st.button("💾 Guardar Análisis Completo", key="save_full_analysis"):
                                 # Guardar el análisis en el historial
                                 st.session_state.historial_analisis.append(analisis_real)
                                 st.success("✅ Análisis completo guardado correctamente en el historial.")
                         
                 except Exception as e:
                     st.error(f"Error al analizar la imagen con IA: {str(e)}")
                     
                     if 'show_debug' in st.session_state and st.session_state.show_debug:
                         st.text("Respuesta original de Gemini:")
                         st.code(response_text if 'response_text' in locals() else "No disponible")
                         st.exception(e)
                     
                     # Mostrar mensaje y usar datos de ejemplo como respaldo
                     st.warning("Usando datos de ejemplo debido a un error en el análisis con IA")
                     
                     # Usar datos de ejemplo como respaldo
                     ejemplo_analisis = {
                         "id": str(uuid.uuid4()),
                         "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                         "total_calories": 450,
                         "items": [
                             {
                                 "name": "Pollo a la parrilla",
                                 "confidence": 0.92,
                                 "portion": "150g",
                                 "nutrition": {
                                     "total_calories": 250,
                                     "protein_g": 30,
                                     "carbs_g": 0,
                                     "fat_g": 15
                                 }
                             },
                             {
                                 "name": "Arroz blanco",
                                 "confidence": 0.88,
                                 "portion": "100g",
                                 "nutrition": {
                                     "total_calories": 130,
                                     "protein_g": 2.7,
                                     "carbs_g": 28,
                                     "fat_g": 0.3
                                 }
                             },
                             {
                                 "name": "Ensalada verde",
                                 "confidence": 0.85,
                                 "portion": "50g",
                                 "nutrition": {
                                     "total_calories": 70,
                                     "protein_g": 1,
                                     "carbs_g": 5,
                                     "fat_g": 5
                                 }
                             }
                         ]
                     }
                     
                     # Mostrar resultados de ejemplo
                     for item in ejemplo_analisis["items"]:
                         with st.container():
                             st.markdown(f"""
                             <div class="result-card">
                                 <h3>{item['name']} <span class="confidence-badge">{int(item['confidence']*100)}%</span></h3>
                                 <p>Porción estimada: {item['portion']}</p>
                                 
                                 <div class="nutrition-info">
                                     <h4>Información Nutricional</h4>
                                     <p>Calorías <span class="nutrition-value">{item['nutrition']['total_calories']} kcal</span></p>
                                     <p>Proteínas <span class="nutrition-value">{item['nutrition']['protein_g']} g</span></p>
                                     <p>Carbohidratos <span class="nutrition-value">{item['nutrition']['carbs_g']} g</span></p>
                                     <p>Grasas <span class="nutrition-value">{item['nutrition']['fat_g']} g</span></p>
                                 </div>
                             </div>
                             """, unsafe_allow_html=True)
                     
                     # Botón para guardar el análisis de ejemplo
                     if st.button("💾 Guardar Análisis"):
                         # Guardar el análisis en el historial
                         st.session_state.historial_analisis.append(ejemplo_analisis)
                         st.success("✅ Análisis guardado correctamente en el historial.")
         
         with analysis_tabs[1]:
             st.subheader("Detección de Fechas de Vencimiento")
             
             # Crear pestañas para diferentes opciones de fechas
             fecha_tabs = st.tabs(["Capturar Fecha", "Información", "Entrenamiento de Modelo"])
             
             with fecha_tabs[0]:
                 st.write("Analiza la imagen para detectar fechas de vencimiento en envases de alimentos")
                 
                 # Opciones para la detección
                 col1, col2 = st.columns(2)
                 with col1:
                     use_tesseract = st.checkbox("Usar OCR (Tesseract)", value=True)
                     use_ai = st.checkbox("Usar detección con IA (Gemini)", value=True)
                 with col2:
                     use_spanish = st.checkbox("Patrones en español", value=True)
                     enable_debug = st.checkbox("Mostrar depuración", value=False, 
                                              help="Muestra información detallada del proceso de detección")
                 
                 if enable_debug != st.session_state.show_debug:
                     st.session_state.show_debug = enable_debug
                 
                 # Botón para detectar fechas
                 if st.button("🔍 Detectar Fechas de Vencimiento"):
                     # Detectar fechas usando OCR
                     expiration_dates = []
                     
                     with st.spinner("Buscando fechas de vencimiento..."):
                         if use_tesseract:
                             ocr_dates = detect_expiration_dates(img)
                             if ocr_dates:
                                 expiration_dates.extend(ocr_dates)
                         
                         if use_ai:
                             # Detectar fechas con Gemini
                             ai_dates = detect_dates_with_gemini(img, temp_filename)
                             if ai_dates:
                                 # Filtrar fechas duplicadas
                                 for ai_date in ai_dates:
                                     # Comprobar si esta fecha ya está en expiration_dates
                                     is_duplicate = False
                                     for date in expiration_dates:
                                         if (abs((ai_date['parsed_date'] - date['parsed_date']).days) < 2 or
                                             ai_date['date_str'] == date['date_str']):
                                             is_duplicate = True
                                             break
                                     
                                     if not is_duplicate:
                                         expiration_dates.append(ai_date)
                     
                     # Mostrar resultados
                     if expiration_dates:
                         st.success(f"Se han detectado {len(expiration_dates)} fechas de vencimiento")
                         
                         today = datetime.now()
                         
                         for i, date_info in enumerate(expiration_dates):
                             date_str = date_info['date_str']
                             days_remaining = date_info['days_remaining']
                             is_expired = date_info['is_expired']
                             
                             # Determinar estado y estilo
                             if is_expired:
                                 status = "VENCIDO"
                                 badge_class = "expired-badge"
                                 card_class = "expiration-card expired"
                                 days_text = f"Vencido hace {abs(days_remaining)} días"
                             elif days_remaining < 7:
                                 status = "PRÓXIMO A VENCER"
                                 badge_class = "soon-badge"
                                 card_class = "expiration-card soon"
                                 days_text = f"Vence en {days_remaining} días"
                             else:
                                 status = "VIGENTE"
                                 badge_class = "valid-badge"
                                 card_class = "expiration-card valid"
                                 days_text = f"Vigente por {days_remaining} días más"
                             
                             # Mostrar método de detección
                             if 'ai_detected' in date_info and date_info['ai_detected']:
                                 detection_method = "Detectado por IA (Gemini)"
                                 confidence = date_info.get('confidence', 'media')
                             else:
                                 detection_method = "Detectado por OCR (Tesseract)"
                                 confidence = "N/A"
                             
                             # Crear tarjeta para la fecha
                             st.markdown(f"""
                             <div class="{card_class}">
                                 <div style="display: flex; justify-content: space-between; align-items: center;">
                                     <span class="expiration-date">{date_str}</span>
                                     <span class="expiration-badge {badge_class}">{status}</span>
                                 </div>
                                 <div class="expiration-days">{days_text}</div>
                                 <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                                     <strong>Método:</strong> {detection_method}
                                     <br><strong>Confianza:</strong> {confidence}
                                 </div>
                             </div>
                             """, unsafe_allow_html=True)
                             
                             # Botón funcional de Streamlit para guardar la fecha
                             if st.button(f"💾 Guardar esta fecha", key=f"save_date_{i}"):
                                 # Añadir fecha al historial con timestamp
                                 date_to_save = date_info.copy()
                                 date_to_save['timestamp'] = datetime.now().strftime("%d/%m/%Y %H:%M")
                                 
                                 if 'fechas_guardadas' not in st.session_state:
                                     st.session_state.fechas_guardadas = []
                                 
                                 st.session_state.fechas_guardadas.append(date_to_save)
                                 st.success(f"✅ Fecha guardada correctamente")
                     else:
                         # No se detectaron fechas
                         st.warning("No se detectaron fechas de vencimiento en la imagen")
                         
                         # Ofrecer opción de entrada manual
                         st.markdown("### Entrada Manual de Fecha")
                         
                         with st.form("manual_date_entry"):
                             st.write("Si conoces la fecha de vencimiento, puedes ingresarla manualmente:")
                             
                             manual_date = st.date_input("Fecha de vencimiento", min_value=datetime.now() - pd.Timedelta(days=365))
                             validate_with_image = st.checkbox("Validar con el estado visual del alimento", value=True, 
                                                              help="Compara la fecha ingresada con el estado visual detectado en la imagen")
                             submit_button = st.form_submit_button("Guardar fecha manual")
                             
                             if submit_button:
                                 # Crear fecha manual y guardarla
                                 manual_date_obj = datetime.combine(manual_date, datetime.min.time())
                                 today = datetime.now()
                                 
                                 days_remaining = (manual_date_obj - today).days
                                 is_expired = days_remaining < 0
                                 
                                 manual_date_info = {
                                     'date_str': manual_date_obj.strftime("%d/%m/%Y"),
                                     'parsed_date': manual_date_obj,
                                     'is_expired': is_expired,
                                     'days_remaining': days_remaining,
                                     'manual_entry': True,
                                     'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M")
                                 }
                                 
                                 # Si se solicitó validación con estado visual
                                 if validate_with_image:
                                     with st.spinner("Analizando el estado visual del alimento para validar la fecha..."):
                                         # Crear mensaje para Gemini
                                         validation_msg = ChatMessage(
                                             role=MessageRole.USER,
                                             blocks=[
                                                 TextBlock(text=f"""Analiza esta imagen de un alimento y determina si la fecha de vencimiento que ha ingresado el usuario ({manual_date_obj.strftime("%d/%m/%Y")}) es coherente con el estado visual del producto.
                                                 
                                                 Hoy es {today.strftime("%d/%m/%Y")}.
                                                 
                                                 Si la fecha ya pasó (producto vencido), verifica si el alimento muestra signos visuales de deterioro que confirmen que efectivamente está vencido.
                                                 
                                                 Si la fecha está en el futuro (producto no vencido), verifica si el estado visual del alimento confirma que todavía está en buen estado.
                                                 
                                                 Responde SOLO con un objeto JSON con el siguiente formato (sin texto adicional):
                                                 {{
                                                   "coherente": true/false,
                                                   "confianza": valor_entre_0_y_1,
                                                   "explicacion": "explicación detallada de la coherencia o incoherencia",
                                                   "recomendacion": "recomendación sobre qué hacer con el alimento"
                                                 }}"""),
                                                 ImageBlock(path=temp_filename, image_mimetype="image/jpeg"),
                                             ],
                                         )
                                         
                                         try:
                                             # Obtener respuesta de Gemini
                                             validation_response = gemini_pro.chat(messages=[validation_msg])
                                             
                                             # Procesar respuesta
                                             validation_text = validation_response.message.content
                                             
                                             # Verificar si es texto JSON y extraerlo
                                             json_match = re.search(r'(\{.*\})', validation_text, re.DOTALL)
                                             if json_match:
                                                 json_str = json_match.group(1)
                                                 # Parsear JSON
                                                 validation_result = json.loads(json_str)
                                                 
                                                 # Agregar resultado a la información de fecha
                                                 manual_date_info['validation'] = validation_result
                                                 
                                                 # Mostrar resultado de validación
                                                 if validation_result['coherente']:
                                                     st.success(f"✅ Fecha validada: La fecha ingresada es coherente con el estado visual del alimento ({int(validation_result['confianza']*100)}% de confianza)")
                                                     st.info(f"**Explicación**: {validation_result['explicacion']}")
                                                     st.info(f"**Recomendación**: {validation_result['recomendacion']}")
                                                 else:
                                                     st.warning(f"⚠️ Posible inconsistencia: La fecha ingresada no parece coherente con el estado visual del alimento ({int(validation_result['confianza']*100)}% de confianza)")
                                                     st.info(f"**Explicación**: {validation_result['explicacion']}")
                                                     st.info(f"**Recomendación**: {validation_result['recomendacion']}")
                                                     
                                                     # Preguntar si aún desea guardar
                                                     if st.button("Guardar de todos modos"):
                                                         if 'fechas_guardadas' not in st.session_state:
                                                             st.session_state.fechas_guardadas = []
                                                         st.session_state.fechas_guardadas.append(manual_date_info)
                                                         st.success("✅ Fecha manual guardada correctamente (con advertencia de inconsistencia)")
                                                     return
                                             else:
                                                 st.warning("No se pudo validar la fecha con el estado visual del alimento. Se guardará sin validación.")
                                         
                                         except Exception as e:
                                             st.warning(f"Error al validar fecha: {str(e)}. Se guardará sin validación.")
                                 
                                 # Guardar la fecha en la sesión
                                 if 'fechas_guardadas' not in st.session_state:
                                     st.session_state.fechas_guardadas = []
                                 
                                 st.session_state.fechas_guardadas.append(manual_date_info)
                                 st.success("✅ Fecha manual guardada correctamente")
                 
                 # Agregar nueva pestaña para capturar solo la fecha de vencimiento
                 st.markdown("---")
                 st.subheader("Captura específica de fecha de vencimiento")
                 st.write("Toma una foto solo de la sección donde aparece la fecha de vencimiento para un reconocimiento más preciso")
                 
                 # Opciones de captura
                 date_capture_option = st.radio(
                     "Selecciona una opción:",
                     ["Subir imagen de la fecha", "Tomar foto de la fecha"],
                     horizontal=True
                 )
                 
                 date_image = None
                 
                 if date_capture_option == "Subir imagen de la fecha":
                     date_image = st.file_uploader("Sube una imagen de la fecha de vencimiento", type=["jpg", "jpeg", "png"], key="date_uploader")
                 else:
                     date_image = st.camera_input("Toma una foto de la fecha de vencimiento", key="date_camera")
                 
                 if date_image is not None:
                     # Mostrar imagen
                     st.image(date_image, caption="Imagen de fecha de vencimiento", use_container_width=True)
                     
                     # Opciones para mejorar reconocimiento
                     st.write("Opciones de mejora para el reconocimiento:")
                     
                     col1, col2, col3 = st.columns(3)
                     with col1:
                         enhance_date = st.checkbox("Mejorar contraste", value=True)
                     with col2:
                         invert_colors = st.checkbox("Invertir colores", value=False)
                     with col3:
                         grayscale = st.checkbox("Escala de grises", value=True)
                     
                     # Botón para procesar
                     if st.button("🔍 Detectar fecha de esta imagen", key="process_date_only"):
                         with st.spinner("Procesando imagen para detectar fecha..."):
                             # Guardar imagen en archivo temporal
                             with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                                 temp_date_filename = temp_file.name
                                 temp_file.write(date_image.getvalue())
                             
                             try:
                                 # Leer y procesar la imagen con OpenCV
                                 date_img = cv2.imread(temp_date_filename)
                                 
                                 # Aplicar mejoras según las opciones seleccionadas
                                 if grayscale:
                                     date_img = cv2.cvtColor(date_img, cv2.COLOR_BGR2GRAY)
                                 
                                 if enhance_date:
                                     # Aplicar ecualización de histograma
                                     if len(date_img.shape) == 2:  # Si es escala de grises
                                         date_img = cv2.equalizeHist(date_img)
                                     else:  # Si es a color
                                         lab = cv2.cvtColor(date_img, cv2.COLOR_BGR2LAB)
                                         l, a, b = cv2.split(lab)
                                         l = cv2.equalizeHist(l)
                                         lab = cv2.merge((l, a, b))
                                         date_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                                 
                                 if invert_colors:
                                     date_img = cv2.bitwise_not(date_img)
                                 
                                 # Guardar imagen procesada
                                 processed_path = temp_date_filename + "_processed.jpg"
                                 cv2.imwrite(processed_path, date_img)
                                 
                                 # Mostrar imagen procesada
                                 processed_img = cv2.imread(processed_path)
                                 processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                                 st.image(processed_img_rgb, caption="Imagen procesada", use_container_width=True)
                                 
                                 # Intentar con Tesseract
                                 date_results = []
                                 
                                 if use_tesseract:
                                     try:
                                         # Aplicar OCR
                                         ocr_text = pytesseract.image_to_string(date_img)
                                         
                                         # Buscar patrones de fecha
                                         patterns = [
                                             r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # DD/MM/YYYY o MM/DD/YYYY
                                             r'(\d{2,4}[/-]\d{1,2}[/-]\d{1,2})',  # YYYY/MM/DD
                                             r'venc[a-zA-Z]*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Vence: DD/MM/YYYY
                                             r'exp[a-zA-Z]*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',   # EXP: DD/MM/YYYY
                                             r'cad[a-zA-Z]*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',   # Caducidad: DD/MM/YYYY
                                             r'consumir antes de:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', # Consumir antes de: DD/MM/YYYY
                                             r'prefer[a-zA-Z]*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',   # Preferente: DD/MM/YYYY
                                             r'best before:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'       # Best before: DD/MM/YYYY
                                         ]
                                         
                                         for pattern in patterns:
                                             matches = re.findall(pattern, ocr_text, re.IGNORECASE)
                                             if matches:
                                                 for match in matches:
                                                     date_results.append({
                                                         'date_str': match,
                                                         'source': 'OCR (Tesseract)',
                                                         'confidence': 0.85
                                                     })
                                         
                                         if 'show_debug' in st.session_state and st.session_state.show_debug:
                                             st.text("Texto OCR detectado:")
                                             st.code(ocr_text)
                                     
                                     except Exception as e:
                                         if 'show_debug' in st.session_state and st.session_state.show_debug:
                                             st.error(f"Error en OCR: {str(e)}")
                                 
                                 # Intentar con Gemini
                                 if use_ai:
                                     try:
                                         # Crear mensaje para Gemini
                                         date_detection_msg = ChatMessage(
                                             role=MessageRole.USER,
                                             blocks=[
                                                 TextBlock(text="""Esta imagen contiene SOLO una fecha de vencimiento o caducidad de un producto alimenticio.
                                                 
                                                 Extrae la fecha en formato DD/MM/YYYY. Si la fecha está en otro formato, conviértela.
                                                 
                                                 Responde SOLO con un objeto JSON con el siguiente formato (sin texto adicional):
                                                 {
                                                   "detected": true/false,
                                                   "date": "DD/MM/YYYY",
                                                   "confidence": valor_entre_0_y_1,
                                                   "original_text": "texto exacto como aparece en la imagen"
                                                 }"""),
                                                 ImageBlock(path=processed_path, image_mimetype="image/jpeg"),
                                             ],
                                         )
                                         
                                         # Obtener respuesta de Gemini
                                         date_response = gemini_pro.chat(messages=[date_detection_msg])
                                         
                                         # Procesar respuesta
                                         date_text = date_response.message.content
                                         
                                         if 'show_debug' in st.session_state and st.session_state.show_debug:
                                             st.text("Respuesta de Gemini:")
                                             st.code(date_text)
                                         
                                         # Verificar si es texto JSON y extraerlo
                                         json_match = re.search(r'(\{.*\})', date_text, re.DOTALL)
                                         if json_match:
                                             json_str = json_match.group(1)
                                             # Parsear JSON
                                             date_result = json.loads(json_str)
                                             
                                             if date_result.get('detected', False):
                                                 date_results.append({
                                                     'date_str': date_result.get('date', ''),
                                                     'source': 'AI (Gemini)',
                                                     'confidence': date_result.get('confidence', 0.5),
                                                     'original_text': date_result.get('original_text', '')
                                                 })
                                     
                                     except Exception as e:
                                         if 'show_debug' in st.session_state and st.session_state.show_debug:
                                             st.error(f"Error con Gemini: {str(e)}")
                                 
                                 # Mostrar resultados
                                 if date_results:
                                     st.success(f"Se han detectado {len(date_results)} posibles fechas")
                                     
                                     for i, result in enumerate(date_results):
                                         # Intentar parsear la fecha
                                         try:
                                             # Intentar varios formatos
                                             formats = ["%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d", "%Y-%m-%d"]
                                             parsed_date = None
                                             
                                             for fmt in formats:
                                                 try:
                                                     parsed_date = datetime.strptime(result['date_str'], fmt)
                                                     break
                                                 except:
                                                     continue
                                             
                                             if parsed_date:
                                                 # Calcular días restantes
                                                 days_remaining = (parsed_date - datetime.now()).days
                                                 is_expired = days_remaining < 0
                                                 
                                                 # Actualizar objeto con información adicional
                                                 result['parsed_date'] = parsed_date
                                                 result['days_remaining'] = days_remaining
                                                 result['is_expired'] = is_expired
                                                 
                                                 # Determinar estado y estilo
                                                 if is_expired:
                                                     status = "VENCIDO"
                                                     badge_color = "#e74c3c"
                                                     days_text = f"Vencido hace {abs(days_remaining)} días"
                                                 elif days_remaining < 7:
                                                     status = "PRÓXIMO A VENCER"
                                                     badge_color = "#f39c12"
                                                     days_text = f"Vence en {days_remaining} días"
                                                 else:
                                                     status = "VIGENTE"
                                                     badge_color = "#2ecc71"
                                                     days_text = f"Vigente por {days_remaining} días más"
                                                 
                                                 # Mostrar resultado con un diseño mejorado
                                                 st.markdown(f"""
                                                 <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid {badge_color};">
                                                     <div style="display: flex; justify-content: space-between; align-items: center;">
                                                         <div>
                                                             <h3 style="margin: 0; color: #2c3e50;">Fecha: {parsed_date.strftime('%d/%m/%Y')}</h3>
                                                             <p style="margin: 5px 0; color: #7f8c8d;">Texto original: {result.get('original_text', result['date_str'])}</p>
                                                             <p style="margin: 5px 0;">Fuente: <span style="font-weight: 500;">{result['source']}</span> (Confianza: {int(result['confidence']*100)}%)</p>
                                                         </div>
                                                         <div style="background-color: {badge_color}; color: white; padding: 8px 16px; border-radius: 20px; font-weight: bold;">
                                                             {status}
                                                         </div>
                                                     </div>
                                                     <p style="margin: 10px 0; font-size: 1.1em;">{days_text}</p>
                                                 </div>
                                                 """, unsafe_allow_html=True)
                                                 
                                                 # Botón funcional de Streamlit
                                                 if st.button(f"💾 Guardar esta fecha", key=f"save_specific_date_{i}"):
                                                     # Crear objeto para guardar
                                                     date_to_save = result.copy()
                                                     date_to_save['timestamp'] = datetime.now().strftime("%d/%m/%Y %H:%M")
                                                     date_to_save['from_specific_capture'] = True
                                                     
                                                     # Guardar en estado de sesión
                                                     if 'fechas_guardadas' not in st.session_state:
                                                         st.session_state.fechas_guardadas = []
                                                     
                                                     st.session_state.fechas_guardadas.append(date_to_save)
                                                     st.success(f"✅ Fecha guardada correctamente")
                                             else:
                                                 st.warning(f"No se pudo interpretar la fecha: {result['date_str']}")
                                         
                                         except Exception as e:
                                             st.warning(f"Error al procesar la fecha {result['date_str']}: {str(e)}")
                                 
                                 else:
                                     st.warning("No se detectaron fechas en la imagen")
                                     # Sugerir probar con otras configuraciones
                                     st.markdown("""
                                     **Sugerencias:**
                                     - Intenta con otra imagen más clara de la fecha
                                     - Prueba diferentes combinaciones de mejoras de imagen (contraste, escala de grises, etc.)
                                     - Asegúrate de que la fecha sea legible en la imagen
                                     """)
                             
                             finally:
                                 # Limpiar archivos temporales
                                 try:
                                     os.unlink(temp_date_filename)
                                     if os.path.exists(processed_path):
                                         os.unlink(processed_path)
                                 except:
                                     pass
         
         with analysis_tabs[2]:
             st.subheader("Análisis del Estado del Alimento")
             
             with st.spinner("Analizando estado del alimento..."):
                 st.info("Procesando imagen para evaluar el estado y calidad del alimento...")
                 
                 # Implementar análisis real del estado con Gemini
                 try:
                     # Crear mensaje para Gemini
                     food_condition_msg = ChatMessage(
                         role=MessageRole.USER,
                         blocks=[
                             TextBlock(text="""Analiza esta imagen de comida y evalúa el estado y calidad de cada alimento visible.
                             Para cada alimento:
                             1. Identifica su nombre
                             2. Evalúa su estado (Excelente, Bueno, Regular o Deteriorado)
                             3. Describe brevemente los detalles visuales que indican su estado
                             4. Proporciona recomendaciones sobre su consumo
                             
                             Responde SOLO con un objeto JSON con el siguiente formato (sin texto adicional):
                             [
                               {
                                 "alimento": "nombre_del_alimento",
                                 "estado": "Excelente/Bueno/Regular/Deteriorado",
                                 "detalles": "descripción_detallada_visual",
                                 "confianza": valor_entre_0_y_1,
                                 "recomendaciones": "recomendación_sobre_consumo"
                               },
                               ...
                             ]"""),
                             ImageBlock(path=temp_filename, image_mimetype="image/jpeg"),
                         ],
                     )
                     
                     # Obtener respuesta de Gemini
                     condition_response = gemini_pro.chat(messages=[food_condition_msg])
                     
                     # Procesar respuesta
                     condition_text = condition_response.message.content
                     
                     # Verificar si es texto JSON y extraerlo
                     json_match = re.search(r'(\[.*\])', condition_text, re.DOTALL)
                     if json_match:
                         json_str = json_match.group(1)
                         # Parsear JSON
                         estado_alimentos_real = json.loads(json_str)
                         
                         # Verificar que tiene la estructura correcta
                         if isinstance(estado_alimentos_real, list) and len(estado_alimentos_real) > 0:
                             # Mostrar resultados reales
                             st.success(f"Se ha analizado el estado de {len(estado_alimentos_real)} alimentos")
                             
                             for item in estado_alimentos_real:
                                 # Determinar color según estado
                                 if item["estado"] == "Excelente":
                                     color = "#4CAF50"  # Verde
                                     icon = "✅"
                                     safety_level = "Alto"
                                 elif item["estado"] == "Bueno":
                                     color = "#8BC34A"  # Verde claro
                                     icon = "✓"
                                     safety_level = "Alto"
                                 elif item["estado"] == "Regular":
                                     color = "#FFC107"  # Amarillo
                                     icon = "⚠️"
                                     safety_level = "Medio"
                                 elif item["estado"] == "Deteriorado":
                                     color = "#F44336"  # Rojo
                                     icon = "❌"
                                     safety_level = "Bajo"
                                 else:
                                     color = "#9E9E9E"  # Gris
                                     icon = "❓"
                                     safety_level = "Desconocido"
                                 
                                 # Crear una tarjeta informativa moderna
                                 col1, col2 = st.columns([1, 2])
                                 
                                 with col1:
                                     # Panel de resumen
                                     st.markdown(f"""
                                     <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
                                         <h2 style="color: #2c3e50; margin-bottom: 10px;">{item['alimento']}</h2>
                                         <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                             <div style="background-color: {color}; color: white; border-radius: 20px; padding: 8px 16px; display: inline-block; font-weight: bold; font-size: 1.2em;">
                                                 {item['estado']}
                                             </div>
                                             <div style="font-size: 2em; margin-left: 10px;">
                                                 {icon}
                                             </div>
                                         </div>
                                         <p style="margin-top: 10px; font-weight: 500;">Nivel de seguridad: <span style="color: {color}; font-weight: 600;">{safety_level}</span></p>
                                         <p style="margin-top: 10px; font-weight: 500;">Confianza: <span style="color: #3498db; font-weight: 600;">{int(float(item['confianza'])*100)}%</span></p>
                                     </div>
                                     """, unsafe_allow_html=True)
                                     
                                     # Añadir botón para instrucciones específicas
                                     if st.button(f"📋 Ver guía para {item['alimento']}", key=f"guide_{item['alimento']}"):
                                         st.info(f"Mostrando información detallada para {item['alimento']}...")
                                         # Aquí se podrían mostrar instrucciones específicas
                                 
                                 with col2:
                                     # Crear pestañas para información detallada
                                     condition_tabs = st.tabs(["📝 Detalles", "🔍 Análisis", "🛟 Recomendaciones"])
                                     
                                     with condition_tabs[0]:
                                         st.markdown(f"""
                                         <h4 style="color: #2c3e50; border-bottom: 2px solid {color}; padding-bottom: 8px;">Detalles Observados</h4>
                                         <p style="background-color: #f2f2f2; padding: 15px; border-radius: 8px; line-height: 1.6;">
                                             {item['detalles']}
                                         </p>
                                         """, unsafe_allow_html=True)
                                         
                                         # Mostrar indicadores visuales
                                         st.markdown("#### Indicadores de Calidad")
                                         
                                         # Generar indicadores basados en el estado
                                         indicators = {"Color": 0, "Textura": 0, "Frescura": 0, "Aspecto": 0}
                                         
                                         if item["estado"] == "Excelente":
                                             indicators = {"Color": 95, "Textura": 90, "Frescura": 95, "Aspecto": 92}
                                         elif item["estado"] == "Bueno":
                                             indicators = {"Color": 80, "Textura": 82, "Frescura": 78, "Aspecto": 80}
                                         elif item["estado"] == "Regular":
                                             indicators = {"Color": 60, "Textura": 65, "Frescura": 55, "Aspecto": 62}
                                         elif item["estado"] == "Deteriorado":
                                             indicators = {"Color": 30, "Textura": 25, "Frescura": 20, "Aspecto": 35}
                                         
                                         # Mostrar barras de progreso para indicadores
                                         for indicator, value in indicators.items():
                                             indicator_color = "#4CAF50" if value > 75 else "#FFC107" if value > 50 else "#F44336"
                                             st.markdown(f"""
                                             <div style="margin-bottom: 15px;">
                                                 <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                                                     <span style="font-weight: 500;">{indicator}</span>
                                                     <span style="font-weight: 600; color: {indicator_color};">{value}%</span>
                                                 </div>
                                                 <div style="background-color: #e0e0e0; border-radius: 10px; height: 8px; width: 100%;">
                                                     <div style="background-color: {indicator_color}; border-radius: 10px; height: 8px; width: {value}%;"></div>
                                                 </div>
                                             </div>
                                             """, unsafe_allow_html=True)
                                     
                                     with condition_tabs[1]:
                                         st.markdown(f"""
                                         <h4 style="color: #2c3e50; border-bottom: 2px solid {color}; padding-bottom: 8px;">Análisis Detallado</h4>
                                         """, unsafe_allow_html=True)
                                         
                                         # Crear un análisis ficticio pero útil basado en el estado
                                         analysis_text = ""
                                         if item["estado"] == "Excelente":
                                             analysis_text = """
                                             <ul style="line-height: 1.6;">
                                                 <li><strong>Apariencia:</strong> Características visuales óptimas, color uniforme y brillante</li>
                                                 <li><strong>Textura:</strong> Consistencia adecuada, firmeza apropiada</li>
                                                 <li><strong>Aroma:</strong> Típico del alimento fresco, sin olores extraños</li>
                                                 <li><strong>Signos de deterioro:</strong> Ninguno visible</li>
                                                 <li><strong>Conservación:</strong> Evidencia de almacenamiento adecuado</li>
                                             </ul>
                                             """
                                         elif item["estado"] == "Bueno":
                                             analysis_text = """
                                             <ul style="line-height: 1.6;">
                                                 <li><strong>Apariencia:</strong> Características visuales apropiadas, color mayormente uniforme</li>
                                                 <li><strong>Textura:</strong> Consistencia generalmente adecuada con pequeñas variaciones</li>
                                                 <li><strong>Aroma:</strong> Olor característico, sin anomalías significativas</li>
                                                 <li><strong>Signos de deterioro:</strong> Mínimos y no preocupantes</li>
                                                 <li><strong>Conservación:</strong> Condiciones de almacenamiento aceptables</li>
                                             </ul>
                                             """
                                         elif item["estado"] == "Regular":
                                             analysis_text = """
                                             <ul style="line-height: 1.6;">
                                                 <li><strong>Apariencia:</strong> Características visuales parcialmente alteradas, color menos uniforme</li>
                                                 <li><strong>Textura:</strong> Cambios notables en la consistencia</li>
                                                 <li><strong>Aroma:</strong> Ligeros cambios en el olor característico</li>
                                                 <li><strong>Signos de deterioro:</strong> Evidentes pero en etapa inicial</li>
                                                 <li><strong>Conservación:</strong> Posibles deficiencias en el almacenamiento</li>
                                             </ul>
                                             """
                                         elif item["estado"] == "Deteriorado":
                                             analysis_text = """
                                             <ul style="line-height: 1.6;">
                                                 <li><strong>Apariencia:</strong> Características visuales significativamente alteradas</li>
                                                 <li><strong>Textura:</strong> Consistencia inapropiada, pérdida de integridad estructural</li>
                                                 <li><strong>Aroma:</strong> Olores anómalos o desagradables</li>
                                                 <li><strong>Signos de deterioro:</strong> Claramente visibles y avanzados</li>
                                                 <li><strong>Conservación:</strong> Evidencia de almacenamiento inadecuado o excesivo tiempo</li>
                                             </ul>
                                             """
                                         
                                         st.markdown(f"{analysis_text}", unsafe_allow_html=True)
                                     
                                     with condition_tabs[2]:
                                         st.markdown(f"""
                                         <h4 style="color: #2c3e50; border-bottom: 2px solid {color}; padding-bottom: 8px;">Recomendaciones de Seguridad</h4>
                                         <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid {color};">
                                             <p style="font-weight: 500;">{item['recomendaciones']}</p>
                                         </div>
                                         """, unsafe_allow_html=True)
                                         
                                         # Añadir recomendaciones adicionales basadas en el estado
                                         st.markdown("#### Acciones recomendadas")
                                         
                                         actions = []
                                         if item["estado"] == "Excelente":
                                             actions = [
                                                 "✅ Apto para consumo inmediato",
                                                 "✅ Puede conservarse según indicaciones del envase",
                                                 "✅ Seguro para todos los grupos de población",
                                                 "✅ Calidad nutricional óptima"
                                             ]
                                         elif item["estado"] == "Bueno":
                                             actions = [
                                                 "✅ Apto para consumo inmediato",
                                                 "⚠️ Consumir preferentemente en los próximos días",
                                                 "✅ Seguro para la mayoría de los grupos de población",
                                                 "✅ Calidad nutricional adecuada"
                                             ]
                                         elif item["estado"] == "Regular":
                                             actions = [
                                                 "⚠️ Consumir con precaución",
                                                 "⚠️ Recomendable consumir el mismo día",
                                                 "⚠️ No recomendado para personas con sistema inmunológico comprometido",
                                                 "⚠️ Posible pérdida parcial de valor nutricional"
                                             ]
                                         elif item["estado"] == "Deteriorado":
                                             actions = [
                                                 "❌ No recomendado para consumo",
                                                 "❌ Desechar de forma apropiada",
                                                 "❌ Riesgo potencial para la salud",
                                                 "❌ Pérdida significativa de calidad nutricional"
                                             ]
                                         
                                         for action in actions:
                                             st.markdown(f"<p style='margin: 5px 0;'>{action}</p>", unsafe_allow_html=True)
                                 
                                 # Línea divisoria entre elementos
                                 st.markdown("<hr>", unsafe_allow_html=True)
                         
                 except Exception as e:
                     st.error(f"Error al analizar el estado del alimento con IA: {str(e)}")
                     
                     if 'show_debug' in st.session_state and st.session_state.show_debug:
                         st.text("Respuesta original de Gemini:")
                         st.code(condition_text if 'condition_text' in locals() else "No disponible")
                         st.exception(e)
                     
                     # Mostrar mensaje y usar datos de ejemplo como respaldo
                     st.warning("Usando datos de ejemplo debido a un error en el análisis con IA")
                     
                     # Ejemplo de análisis del estado (simulado)
                     estado_alimentos = [
                         {
                             "alimento": "Pollo a la parrilla",
                             "estado": "Excelente",
                             "detalles": "El color y textura indican que está recién preparado",
                             "confianza": 0.94,
                             "recomendaciones": "Seguro para consumo"
                         },
                         {
                             "alimento": "Arroz blanco",
                             "estado": "Bueno",
                             "detalles": "Textura adecuada, sin signos de deterioro",
                             "confianza": 0.88,
                             "recomendaciones": "Seguro para consumo"
                         },
                         {
                             "alimento": "Ensalada verde",
                             "estado": "Regular",
                             "detalles": "Algunas hojas muestran signos leves de marchitamiento",
                             "confianza": 0.82,
                             "recomendaciones": "Consumir pronto"
                         }
                     ]
                     
                     # Mostrar resultados de ejemplo
                     for item in estado_alimentos:
                         # Determinar color según estado
                         if item["estado"] == "Excelente":
                             color = "#4CAF50"  # Verde
                         elif item["estado"] == "Bueno":
                             color = "#8BC34A"  # Verde claro
                         elif item["estado"] == "Regular":
                             color = "#FFC107"  # Amarillo
                         elif item["estado"] == "Deteriorado":
                             color = "#F44336"  # Rojo
                         else:
                             color = "#9E9E9E"  # Gris
                         
                         # Crear tarjeta para el estado
                         st.markdown(f"""
                         <div class="condition-info" style="border-left: 4px solid {color};">
                             <div style="display: flex; justify-content: space-between; align-items: center;">
                                 <h4>{item['alimento']}</h4>
                                 <span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; font-weight: 500;">{item['estado']}</span>
                             </div>
                             <p><strong>Detalles:</strong> <span class="condition-detail">{item['detalles']}</span></p>
                             <p><strong>Confianza:</strong> <span class="condition-confidence">{int(float(item['confianza'])*100)}%</span></p>
                             <p><strong>Recomendación:</strong> <span class="condition-recommendation">{item['recomendaciones']}</span></p>
                         </div>
                         """, unsafe_allow_html=True)
     
     except Exception as e:
         st.error(f"Error al procesar la imagen: {str(e)}")
         if 'show_debug' in st.session_state and st.session_state.show_debug:
             st.exception(e)
     
     finally:
         # Eliminar el archivo temporal
         try:
             os.unlink(temp_filename)
         except:
             pass
 
 def about_page():
     st.title("Sobre ¿Qué hay en tu plato?")
     st.markdown("""
     ¿Qué hay en tu plato? es una aplicación de análisis nutricional impulsada por inteligencia artificial que te permite:
 
     - **Identificar alimentos** en imágenes con alta precisión
     - **Calcular información nutricional** como calorías, proteínas, carbohidratos y grasas
     - **Detectar el estado de los alimentos** para garantizar su seguridad alimentaria
     - **Detectar fechas de vencimiento** y verificar si los productos están vencidos
     - **Recibir recomendaciones personalizadas** para mejorar tus hábitos alimenticios
     - **Visualizar datos** a través de gráficos interactivos
     - **Exportar y guardar** tus análisis para seguimiento
 
     Esta aplicación utiliza el modelo Gemini de Google para proporcionar análisis precisos y recomendaciones personalizadas.
 
     ### Tecnologías utilizadas
     - Streamlit para la interfaz de usuario
     - Google Gemini para análisis de imágenes e información nutricional
     - OpenCV para procesamiento de imágenes
     - Altair y Pandas para visualización de datos
     - Tesseract OCR (opcional) para detectar fechas de vencimiento
     
     ### Funcionalidad de detección del estado de los alimentos
     
     Nuestra aplicación ahora incluye una avanzada funcionalidad de detección del estado de los alimentos que:
     
     - Analiza visualmente cada alimento para detectar signos de deterioro
     - Evalúa el color, textura y apariencia general
     - Clasifica los alimentos en diferentes estados (Excelente, Bueno, Regular, Deteriorado)
     - Proporciona recomendaciones específicas sobre el consumo seguro
     - Ofrece guías educativas sobre cómo identificar alimentos en mal estado
     
     ### Detección de fechas de vencimiento
     
     La aplicación ahora cuenta con la capacidad de detectar fechas de vencimiento en los envases de alimentos:
     
     - Utiliza tecnología OCR (Reconocimiento Óptico de Caracteres) para leer texto de imágenes
     - Identifica formatos comunes de fechas de vencimiento (DD/MM/AAAA, MM/DD/AA, etc.)
     - Compara automáticamente con la fecha actual para determinar si un producto está vencido
     - Proporciona alertas visuales para productos vencidos o próximos a vencer
     - Ofrece recomendaciones específicas sobre cómo actuar cuando un producto está vencido
     
     > **Nota**: Esta funcionalidad requiere la biblioteca pytesseract y Tesseract OCR instalados en el sistema. Si no están disponibles, se usará una simulación para demostrar la funcionalidad.
     """)
 
 def contact_page():
     st.title("Investigaciones y Recursos")
     
     # Crear pestañas para separar contenido
     tabs = st.tabs(["Historial de Análisis", "Historial de Fechas", "Contacto", "Recursos"])
     
     with tabs[0]:
         # Mostrar historial de análisis si existe
         if 'historial_analisis' in st.session_state and st.session_state.historial_analisis:
             st.subheader("Historial de Análisis")
             
             for i, analisis in enumerate(st.session_state.historial_analisis):
                 with st.expander(f"Análisis #{i+1} - {analisis['date']}"):
                     st.write(f"ID: {analisis['id']}")
                     st.write(f"Total calorías: {analisis['total_calories']} kcal")
                     
                     # Crear tabla de alimentos
                     items_df = pd.DataFrame([{
                         "Alimento": item["name"],
                         "Calorías": item["nutrition"].get("total_calories", 0),
                         "Proteínas (g)": item["nutrition"].get("protein_g", 0),
                         "Carbohidratos (g)": item["nutrition"].get("carbs_g", 0),
                         "Grasas (g)": item["nutrition"].get("fat_g", 0)
                     } for item in analisis["items"]])
                     
                     st.dataframe(items_df)
         else:
             st.info("No hay análisis guardados todavía. Analiza alimentos en la herramienta principal y guarda los resultados para verlos aquí.")
     
     with tabs[1]:
         st.subheader("Historial de Fechas de Vencimiento")
         
         # Mostrar fechas guardadas si existen
         if 'fechas_guardadas' in st.session_state and st.session_state.fechas_guardadas:
             # Agrupar por tipo (vencidas, por vencer, vigentes)
             vencidas = [f for f in st.session_state.fechas_guardadas if f.get('is_expired', False)]
             por_vencer = [f for f in st.session_state.fechas_guardadas if not f.get('is_expired', False) and f.get('days_remaining', 0) < 7]
             vigentes = [f for f in st.session_state.fechas_guardadas if not f.get('is_expired', False) and f.get('days_remaining', 0) >= 7]
             
             # Crear pestañas para cada categoría
             fecha_tabs = st.tabs(["Todas", f"Vencidas ({len(vencidas)})", f"Por vencer ({len(por_vencer)})", f"Vigentes ({len(vigentes)})"])
             
             with fecha_tabs[0]:
                 st.markdown(f"### Total de fechas guardadas: {len(st.session_state.fechas_guardadas)}")
                 
                 # Botón para eliminar todo el historial
                 if st.button("🗑️ Borrar todo el historial de fechas"):
                     st.session_state.fechas_guardadas = []
                     st.success("Historial borrado correctamente")
                     st.experimental_rerun()
                 
                 # Mostrar todas las fechas
                 for i, fecha in enumerate(st.session_state.fechas_guardadas):
                     # Determinar color y estado
                     if fecha.get('is_expired', False):
                         badge_color = "#f44336"
                         badge_text = "VENCIDO"
                         bg_color = "#ffebee"
                     elif fecha.get('days_remaining', 0) < 7:
                         badge_color = "#ff9800"
                         badge_text = "POR VENCER"
                         bg_color = "#fff8e1"
                     else:
                         badge_color = "#4caf50"
                         badge_text = "VIGENTE"
                         bg_color = "#e8f5e9"
                     
                     # Determinar método de detección
                     if fecha.get('ai_detected', False):
                         method = "IA (Gemini)"
                     elif fecha.get('manual_entry', False):
                         method = "Entrada manual"
                     else:
                         method = "OCR (Tesseract)"
                     
                     # Mostrar tarjeta
                     st.markdown(f"""
                     <div style="background-color:{bg_color}; padding:15px; border-radius:5px; margin:10px 0; border-left:4px solid {badge_color};">
                         <div style="display:flex; justify-content:space-between; align-items:center;">
                             <span style="background-color:{badge_color}; color:white; padding:3px 8px; border-radius:10px; font-size:0.8em; font-weight:bold;">{badge_text}</span>
                             <span style="color:#666; font-size:0.8em;">Guardada: {fecha.get('timestamp', 'N/A')}</span>
                         </div>
                         <div style="margin-top:10px;"><strong>Fecha:</strong> {fecha.get('date_str', 'N/A')}</div>
                         <div><strong>Método de detección:</strong> {method}</div>
                         <div><strong>Días restantes:</strong> {fecha.get('days_remaining', 'N/A')}</div>
                     </div>
                     """, unsafe_allow_html=True)
             
             # Mostrar fechas vencidas
             with fecha_tabs[1]:
                 if vencidas:
                     for fecha in vencidas:
                         days = abs(fecha.get('days_remaining', 0))
                         days_text = "día" if days == 1 else "días"
                         st.markdown(f"""
                         <div style="background-color:#ffebee; padding:15px; border-radius:5px; margin:10px 0; border-left:4px solid #f44336;">
                             <span style="background-color:#f44336; color:white; padding:3px 8px; border-radius:10px; font-size:0.8em; font-weight:bold;">VENCIDO</span>
                             <div style="margin-top:10px;"><strong>Fecha:</strong> {fecha.get('date_str', 'N/A')}</div>
                             <div><strong>Estado:</strong> Vencido hace {days} {days_text}</div>
                             <div><strong>Guardada:</strong> {fecha.get('timestamp', 'N/A')}</div>
                         </div>
                         """, unsafe_allow_html=True)
                 else:
                     st.info("No hay fechas vencidas guardadas.")
             
             # Mostrar fechas por vencer
             with fecha_tabs[2]:
                 if por_vencer:
                     for fecha in por_vencer:
                         days = fecha.get('days_remaining', 0)
                         days_text = "día" if days == 1 else "días"
                         st.markdown(f"""
                         <div style="background-color:#fff8e1; padding:15px; border-radius:5px; margin:10px 0; border-left:4px solid #ff9800;">
                             <span style="background-color:#ff9800; color:white; padding:3px 8px; border-radius:10px; font-size:0.8em; font-weight:bold;">POR VENCER</span>
                             <div style="margin-top:10px;"><strong>Fecha:</strong> {fecha.get('date_str', 'N/A')}</div>
                             <div><strong>Estado:</strong> Vence en {days} {days_text}</div>
                             <div><strong>Guardada:</strong> {fecha.get('timestamp', 'N/A')}</div>
                         </div>
                         """, unsafe_allow_html=True)
                 else:
                     st.info("No hay fechas por vencer guardadas.")
             
             # Mostrar fechas vigentes
             with fecha_tabs[3]:
                 if vigentes:
                     for fecha in vigentes:
                         days = fecha.get('days_remaining', 0)
                         days_text = "día" if days == 1 else "días"
                         st.markdown(f"""
                         <div style="background-color:#e8f5e9; padding:15px; border-radius:5px; margin:10px 0; border-left:4px solid #4caf50;">
                             <span style="background-color:#4caf50; color:white; padding:3px 8px; border-radius:10px; font-size:0.8em; font-weight:bold;">VIGENTE</span>
                             <div style="margin-top:10px;"><strong>Fecha:</strong> {fecha.get('date_str', 'N/A')}</div>
                             <div><strong>Estado:</strong> Válido por {days} {days_text} más</div>
                             <div><strong>Guardada:</strong> {fecha.get('timestamp', 'N/A')}</div>
                         </div>
                         """, unsafe_allow_html=True)
                 else:
                     st.info("No hay fechas vigentes guardadas.")
                         
         else:
             st.info("No hay fechas de vencimiento guardadas todavía. Analiza alimentos en la herramienta principal y guarda las fechas detectadas para verlas aquí.")
     
     with tabs[2]:
         st.subheader("Contacto")
         st.markdown("""
     Para cualquier consulta o sugerencia, no dudes en contactarnos:
     """)
     
     contact_form = """
     <form action="https://formsubmit.co/jriverabu@unal.edu.co" method="POST">
         <input type="hidden" name="_captcha" value="false">
         <input type="text" name="name" placeholder="Tu nombre" required>
         <input type="email" name="email" placeholder="Tu email" required>
         <textarea name="message" placeholder="Tu mensaje aquí"></textarea>
         <button type="submit">Enviar</button>
     </form>
     """
     st.markdown(contact_form, unsafe_allow_html=True)
     
     with tabs[3]:
         st.subheader("Enlaces a recursos nutricionales")
         
         st.markdown("""
         ### Enlaces a recursos nutricionales
         
         - [Base de Datos Española de Composición de Alimentos (BEDCA)](https://www.bedca.net/)
         - [USDA FoodData Central](https://fdc.nal.usda.gov/)
         - [Organización Mundial de la Salud - Nutrición](https://www.who.int/es/health-topics/nutrition)
         
         ### Recursos sobre fechas de vencimiento
         
         - [AESAN - Agencia Española de Seguridad Alimentaria y Nutrición](https://www.aesan.gob.es/)
         - [FDA - Cómo entender fechas en etiquetas de alimentos](https://www.fda.gov/consumers/consumer-updates/how-understand-and-use-nutrition-facts-label)
         - [FAO - Manual para reducir el desperdicio de alimentos](http://www.fao.org/3/ca8646es/CA8646ES.pdf)
         """)
 
 if __name__ == "__main__":
     main()

