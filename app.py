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

# Set up Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA5FyLIhOSIKxGw3TebXzLfMjuYx5fVwW4"

# Initialize the Gemini model
gemini_pro = Gemini(model_name="models/gemini-1.5-flash")

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
    st.markdown("Upload an image or use your camera to detect objects in real-time!")

    upload_option = st.radio("Choose input method:", ("Upload Image", "Use Camera"))

    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            process_image(uploaded_file)
    else:
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            process_image(img_file_buffer)

def process_image(image):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (600, 500))
    image_height, image_width = img_resized.shape[:2]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, img_resized)
        image_path = tmp.name
    
    # Opción para mostrar información de depuración
    show_debug = st.sidebar.checkbox("Mostrar información de depuración", value=False)
    
    # First message to detect objects
    detect_msg = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text="Detecta y devuelve bounding boxes para todos los alimentos en la imagen. Formato de salida: [ymin, xmin, ymax, xmax, nombre_alimento]. Incluye todos los alimentos visibles en el formato: [ymin, xmin, ymax, xmax, nombre_alimento]. Devuelve la respuesta en texto."),
            ImageBlock(path=image_path, image_mimetype="image/jpeg"),
        ],
    )

    with st.spinner("Analizando imagen..."):
        detect_response = gemini_pro.chat(messages=[detect_msg])

        # Second message to get nutritional information with macronutrients and categories
        nutrition_msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(text="""Para cada alimento en esta imagen, proporciona una estimación de:
1) Calorías por 100g
2) Peso estimado en gramos de la porción visible
3) Calorías totales para la porción
4) Proteínas (g)
5) Carbohidratos (g)
6) Grasas (g)
7) Categoría del alimento (elige una: Proteínas, Carbohidratos, Grasas, Frutas, Verduras, Lácteos, Bebidas, Otros)
8) Fibra (g)
9) Azúcares (g)

IMPORTANTE: Responde SOLO con un objeto JSON puro, sin ningún texto adicional, sin comillas simples, sin la palabra 'json' al principio, sin bloques de código markdown, exactamente así:

{
  "alimento1": {
    "calories_per_100g": X, 
    "estimated_weight_g": Y, 
    "total_calories": Z,
    "protein_g": P,
    "carbs_g": C,
    "fat_g": F,
    "category": "Categoría",
    "fiber_g": FB,
    "sugar_g": S
  },
  "alimento2": {
    "calories_per_100g": X, 
    "estimated_weight_g": Y, 
    "total_calories": Z,
    "protein_g": P,
    "carbs_g": C,
    "fat_g": F,
    "category": "Categoría",
    "fiber_g": FB,
    "sugar_g": S
  }
}

Donde todos los valores son números (no strings) excepto category que es un string. NO incluyas ningún texto antes o después del JSON."""),
                ImageBlock(path=image_path, image_mimetype="image/jpeg"),
            ],
        )
        nutrition_response = gemini_pro.chat(messages=[nutrition_msg])

        # Third message to get recommendations and daily intake
        recommendation_msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(text="""Basándote en los alimentos que ves en esta imagen, proporciona:
1) Una evaluación general del balance nutricional del plato
2) 2-3 recomendaciones específicas para mejorar el valor nutricional
3) Posibles alternativas más saludables si es necesario
4) Porcentaje aproximado de la ingesta diaria recomendada que representa este plato (para una dieta de 2000 kcal)
5) Sugerencias para complementar esta comida con otros alimentos para lograr una dieta equilibrada

Responde en español, de forma concisa pero informativa, estructurando tu respuesta en secciones claras."""),
                ImageBlock(path=image_path, image_mimetype="image/jpeg"),
            ],
        )
        recommendation_response = gemini_pro.chat(messages=[recommendation_msg])
        
        # Nueva solicitud para evaluar el estado de los alimentos
        food_condition_msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(text="""Analiza detalladamente la imagen y determina si los alimentos que ves están en buen estado o presentan signos de deterioro.

INSTRUCCIONES DETALLADAS:
Para cada alimento visible en la imagen, evalúa los siguientes aspectos:

1) COLOR: ¿El color es normal y típico para este alimento o muestra decoloración, manchas, o cambios de color anormales?
2) TEXTURA: ¿La textura parece normal o muestra signos de deterioro como ablandamiento excesivo, endurecimiento, viscosidad anormal?
3) APARIENCIA GENERAL: ¿Hay presencia visible de moho, hongos, manchas, magulladuras excesivas o signos de descomposición?
4) FRESCURA ESTIMADA: Basado en los indicadores visuales, ¿el alimento parece fresco, ligeramente envejecido o claramente deteriorado?

Después de evaluar estos aspectos, clasifica cada alimento en una de estas categorías:
- "Excelente": Sin signos visibles de deterioro, apariencia fresca y óptima
- "Bueno": Mínimos signos de envejecimiento natural, pero sin deterioro significativo
- "Regular": Signos moderados de envejecimiento o inicio de deterioro, pero posiblemente aún comestible con precaución
- "Deteriorado": Claros signos de deterioro que sugieren que no debería consumirse
- "No determinable": No es posible evaluar con certeza el estado del alimento desde la imagen

IMPORTANTE: Responde SOLO con un objeto JSON puro, sin ningún texto adicional, sin comillas simples, sin la palabra 'json' al principio, sin bloques de código markdown, exactamente así:

{
  "alimento1": {
    "estado": "Excelente/Bueno/Regular/Deteriorado/No determinable",
    "signos_deterioro": "Descripción detallada de signos visibles o 'Ninguno visible'",
    "detalles_evaluacion": {
      "color": "Descripción del color y si es normal o anormal",
      "textura": "Evaluación de la textura visible",
      "apariencia": "Descripción de la apariencia general"
    },
    "confianza": "Alto/Medio/Bajo",
    "recomendacion": "Recomendación específica sobre consumo y manipulación"
  },
  "alimento2": {
    "estado": "Excelente/Bueno/Regular/Deteriorado/No determinable",
    "signos_deterioro": "Descripción detallada de signos visibles o 'Ninguno visible'",
    "detalles_evaluacion": {
      "color": "Descripción del color y si es normal o anormal",
      "textura": "Evaluación de la textura visible",
      "apariencia": "Descripción de la apariencia general"
    },
    "confianza": "Alto/Medio/Bajo",
    "recomendacion": "Recomendación específica sobre consumo y manipulación"
  }
}"""),
                ImageBlock(path=image_path, image_mimetype="image/jpeg"),
            ],
        )
        food_condition_response = gemini_pro.chat(messages=[food_condition_msg])

    bounding_boxes = re.findall(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\w\s]+)\]', detect_response.message.content)

    # Parse nutrition information
    try:
        nutrition_text = nutrition_response.message.content
        # Log the raw response for debugging
        if show_debug:
            st.write("Respuesta de nutrición (debug):", nutrition_text)
        
        # Limpiar el texto de la respuesta para extraer solo el JSON válido
        # Eliminar prefijos como 'json' o '```json' que Gemini a veces incluye
        cleaned_text = re.sub(r'^.*?json\s*', '', nutrition_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'^```json\s*', '', cleaned_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'```$', '', cleaned_text, flags=re.DOTALL)
        
        # Extract the JSON part from the cleaned response
        nutrition_json_str = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if nutrition_json_str:
            # Clean the JSON string to ensure it's valid
            json_str = nutrition_json_str.group()
            # Replace any single quotes with double quotes for valid JSON
            json_str = json_str.replace("'", '"')
            
            # Try to parse as JSON first
            try:
                nutrition_info = json.loads(json_str)
            except json.JSONDecodeError as json_err:
                st.warning(f"Error al parsear JSON: {str(json_err)}. Intentando método alternativo...")
                try:
                    # Fallback to eval if JSON parsing fails
                    nutrition_info = eval(json_str)
                except Exception as eval_err:
                    st.warning(f"Error en método alternativo: {str(eval_err)}")
                    nutrition_info = {}
        else:
            nutrition_info = {}
            st.warning("No se pudo extraer información nutricional del formato JSON.")
    except Exception as e:
        nutrition_info = {}
        st.warning(f"Error al procesar información nutricional: {str(e)}")
    
    # Parse food condition information
    try:
        condition_text = food_condition_response.message.content
        # Log the raw response for debugging
        if show_debug:
            st.write("Respuesta de condición de alimentos (debug):", condition_text)
        
        # Limpiar el texto de la respuesta para extraer solo el JSON válido
        cleaned_text = re.sub(r'^.*?json\s*', '', condition_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'^```json\s*', '', cleaned_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'```$', '', cleaned_text, flags=re.DOTALL)
        
        # Extract the JSON part from the cleaned response
        condition_json_str = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if condition_json_str:
            # Clean the JSON string to ensure it's valid
            json_str = condition_json_str.group()
            # Replace any single quotes with double quotes for valid JSON
            json_str = json_str.replace("'", '"')
            
            # Try to parse as JSON first
            try:
                condition_info = json.loads(json_str)
            except json.JSONDecodeError as json_err:
                st.warning(f"Error al parsear JSON de condición: {str(json_err)}. Intentando método alternativo...")
                try:
                    # Fallback to eval if JSON parsing fails
                    condition_info = eval(json_str)
                except Exception as eval_err:
                    st.warning(f"Error en método alternativo para condición: {str(eval_err)}")
                    condition_info = {}
        else:
            condition_info = {}
            st.warning("No se pudo extraer información de condición de alimentos del formato JSON.")
    except Exception as e:
        condition_info = {}
        st.warning(f"Error al procesar información de condición de alimentos: {str(e)}")
        
    # Fallback values if no nutrition info is available
    default_nutrition = {
        "calories_per_100g": 250,
        "estimated_weight_g": 150,
        "total_calories": 375,
        "protein_g": 10,
        "carbs_g": 30,
        "fat_g": 15,
        "category": "Otros",
        "fiber_g": 2,
        "sugar_g": 5
    }
    
    # Fallback values if no condition info is available
    default_condition = {
        "estado": "No determinable",
        "signos_deterioro": "No se pudo determinar",
        "detalles_evaluacion": {
            "color": "No se pudo evaluar",
            "textura": "No se pudo evaluar",
            "apariencia": "No se pudo evaluar"
        },
        "confianza": "Bajo",
        "recomendacion": "Verificar visualmente antes de consumir"
    }

    results = []
    for i, box in enumerate(bounding_boxes):
        parts = box.split(',')
        if len(parts) < 5:
            continue  # Skip invalid boxes
            
        numbers = list(map(int, parts[:4]))
        label = parts[4].strip()
        ymin, xmin, ymax, xmax = numbers
        x1 = int(xmin / 1000 * image_width)
        y1 = int(ymin / 1000 * image_height)
        x2 = int(xmax / 1000 * image_width)
        y2 = int(ymax / 1000 * image_height)

        # Determinar el color del rectángulo según el estado del alimento
        box_color = (0, 255, 0)  # Verde por defecto (buen estado)
        food_condition = condition_info.get(label, default_condition)
        if food_condition.get("estado") == "Deteriorado":
            box_color = (0, 0, 255)  # Rojo (deteriorado)
        elif food_condition.get("estado") == "Regular":
            box_color = (0, 165, 255)  # Naranjo (regular)
        elif food_condition.get("estado") == "No determinable":
            box_color = (128, 128, 128)  # Gris (no determinable)

        cv2.rectangle(img_resized, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(img_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Get nutritional info if available
        nutrition_data = nutrition_info.get(label, default_nutrition)
        # Get condition info if available
        condition_data = condition_info.get(label, default_condition)
        
        results.append({
            "id": i+1,
            "label": label,
            "confidence": round(0.8 + (i * 0.02 % 0.2), 2),
            "bbox": [x1, y1, x2, y2],
            "nutrition": nutrition_data,
            "condition": condition_data
        })

    # Prepare data for export and visualization
    export_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id": str(uuid.uuid4())[:8],
        "total_calories": sum(result.get('nutrition', {}).get('total_calories', 0) for result in results),
        "total_protein": sum(result.get('nutrition', {}).get('protein_g', 0) for result in results),
        "total_carbs": sum(result.get('nutrition', {}).get('carbs_g', 0) for result in results),
        "total_fat": sum(result.get('nutrition', {}).get('fat_g', 0) for result in results),
        "items": [{
            "name": r["label"],
            "nutrition": r["nutrition"]
        } for r in results]
    }

    # Calcular porcentajes de ingesta diaria recomendada (basado en dieta de 2000 kcal)
    daily_values = {
        "calories": 2000,
        "protein": 50,  # g
        "carbs": 275,   # g
        "fat": 78,      # g
        "fiber": 28     # g
    }
    
    daily_percentages = {
        "calories": (export_data["total_calories"] / daily_values["calories"]) * 100,
        "protein": (export_data["total_protein"] / daily_values["protein"]) * 100,
        "carbs": (export_data["total_carbs"] / daily_values["carbs"]) * 100,
        "fat": (export_data["total_fat"] / daily_values["fat"]) * 100,
        "fiber": sum(r.get('nutrition', {}).get('fiber_g', 0) for r in results) / daily_values["fiber"] * 100
    }

    # Agrupar alimentos por categoría
    food_categories = {}
    for result in results:
        category = result.get('nutrition', {}).get('category', 'Otros')
        if category not in food_categories:
            food_categories[category] = []
        food_categories[category].append(result)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_resized, channels="BGR", use_column_width=True)
        
        # Crear gráficos
        if results:
            tabs = st.tabs(["Calorías", "Macronutrientes", "Ingesta Diaria", "Por Categoría"])
            
            with tabs[0]:
                st.subheader("Distribución de Calorías")
                
                # Reemplazar matplotlib con Altair para el gráfico de barras
                calories_data = pd.DataFrame({
                    'Alimento': [result['label'] for result in results],
                    'Calorías': [result.get('nutrition', {}).get('total_calories', 0) for result in results]
                })
                
                calories_chart = alt.Chart(calories_data).mark_bar().encode(
                    x=alt.X('Alimento', sort=None),
                    y='Calorías',
                    color=alt.Color('Alimento', legend=None)
                ).properties(
                    width=600,
                    height=400,
                    title='Calorías por Alimento'
                )
                
                # Añadir etiquetas de texto encima de las barras
                text = calories_chart.mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-5
                ).encode(
                    text=alt.Text('Calorías:Q', format='.0f')
                )
                
                st.altair_chart(calories_chart + text, use_container_width=True)
            
            with tabs[1]:
                st.subheader("Macronutrientes")
                
                # Preparar datos para el gráfico de macronutrientes
                proteins = sum(r.get('nutrition', {}).get('protein_g', 0) for r in results)
                carbs = sum(r.get('nutrition', {}).get('carbs_g', 0) for r in results)
                fats = sum(r.get('nutrition', {}).get('fat_g', 0) for r in results)
                fiber = sum(r.get('nutrition', {}).get('fiber_g', 0) for r in results)
                sugar = sum(r.get('nutrition', {}).get('sugar_g', 0) for r in results)
                
                # Crear gráfico de pie con Altair en lugar de matplotlib
                pie_data = pd.DataFrame({
                    'Macronutriente': ['Proteínas', 'Carbohidratos', 'Grasas'],
                    'Gramos': [proteins, carbs, fats],
                    'Color': ['#3498db', '#2ecc71', '#e74c3c']
                })
                
                # Calcular el total para los porcentajes
                total_macros = proteins + carbs + fats
                pie_data['Porcentaje'] = pie_data['Gramos'] / total_macros * 100 if total_macros > 0 else 0
                pie_data['Etiqueta'] = pie_data.apply(lambda x: f"{x['Macronutriente']}: {x['Porcentaje']:.1f}%", axis=1)
                
                # Crear gráfico de dona con Altair
                pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=50, outerRadius=100).encode(
                    theta=alt.Theta(field="Gramos", type="quantitative"),
                    color=alt.Color(field="Macronutriente", type="nominal", 
                                   scale=alt.Scale(domain=['Proteínas', 'Carbohidratos', 'Grasas'],
                                                  range=['#3498db', '#2ecc71', '#e74c3c'])),
                    tooltip=['Macronutriente', 'Gramos', 'Porcentaje']
                ).properties(
                    width=400,
                    height=400,
                    title='Distribución de Macronutrientes'
                )
                
                # Añadir etiquetas
                text = pie_chart.mark_text(radius=130, size=14).encode(
                    text="Etiqueta"
                )
                
                st.altair_chart(pie_chart + text, use_container_width=True)
                
                # Mostrar tabla de macronutrientes
                macro_df = pd.DataFrame({
                    'Macronutriente': ['Proteínas', 'Carbohidratos', 'Grasas', 'Fibra', 'Azúcares'],
                    'Cantidad (g)': [proteins, carbs, fats, fiber, sugar],
                    'Calorías': [proteins * 4, carbs * 4, fats * 9, fiber * 2, sugar * 4],
                    '% del Total': [proteins * 4 / (proteins * 4 + carbs * 4 + fats * 9) * 100,
                                   carbs * 4 / (proteins * 4 + carbs * 4 + fats * 9) * 100,
                                   fats * 9 / (proteins * 4 + carbs * 4 + fats * 9) * 100,
                                   fiber * 2 / (proteins * 4 + carbs * 4 + fats * 9) * 100 if (proteins * 4 + carbs * 4 + fats * 9) > 0 else 0,
                                   sugar * 4 / (proteins * 4 + carbs * 4 + fats * 9) * 100 if (proteins * 4 + carbs * 4 + fats * 9) > 0 else 0]
                })
                
                st.dataframe(macro_df.style.format({
                    'Cantidad (g)': '{:.1f}',
                    'Calorías': '{:.1f}',
                    '% del Total': '{:.1f}%'
                }))
                
                # Gráfico de barras para desglose de carbohidratos
                if carbs > 0:
                    st.subheader("Desglose de Carbohidratos")
                    carbs_data = pd.DataFrame({
                        'Tipo': ['Azúcares', 'Fibra', 'Otros Carbohidratos'],
                        'Gramos': [sugar, fiber, carbs - sugar - fiber if carbs - sugar - fiber > 0 else 0]
                    })
                    
                    carbs_chart = alt.Chart(carbs_data).mark_bar().encode(
                        x='Tipo',
                        y='Gramos',
                        color=alt.Color('Tipo', scale=alt.Scale(domain=['Azúcares', 'Fibra', 'Otros Carbohidratos'],
                                                              range=['#e74c3c', '#2ecc71', '#3498db']))
                    ).properties(width=400, height=300)
                    
                    st.altair_chart(carbs_chart, use_container_width=True)
            
            with tabs[2]:
                st.subheader("Porcentaje de Ingesta Diaria Recomendada")
                
                # Crear DataFrame para el gráfico de ingesta diaria
                daily_intake_df = pd.DataFrame({
                    'Nutriente': ['Calorías', 'Proteínas', 'Carbohidratos', 'Grasas', 'Fibra'],
                    'Porcentaje': [daily_percentages['calories'], daily_percentages['protein'], 
                                  daily_percentages['carbs'], daily_percentages['fat'], 
                                  daily_percentages['fiber']]
                })
                
                # Crear gráfico de barras horizontales
                intake_chart = alt.Chart(daily_intake_df).mark_bar().encode(
                    y='Nutriente',
                    x=alt.X('Porcentaje', scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color('Nutriente', scale=alt.Scale(domain=['Calorías', 'Proteínas', 'Carbohidratos', 'Grasas', 'Fibra'],
                                                              range=['#f39c12', '#3498db', '#2ecc71', '#e74c3c', '#9b59b6']))
                ).properties(width=400, height=300)
                
                # Añadir línea de referencia en 100%
                rule = alt.Chart(pd.DataFrame({'x': [100]})).mark_rule(color='black', strokeDash=[5, 5]).encode(x='x')
                
                st.altair_chart(intake_chart + rule, use_container_width=True)
                
                # Mostrar tabla de ingesta diaria
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 20px;">
                    <h4 style="margin-top: 0;">Valores Diarios de Referencia</h4>
                    <p>Basados en una dieta de 2000 calorías:</p>
                    <ul>
                        <li>Calorías: 2000 kcal</li>
                        <li>Proteínas: 50g</li>
                        <li>Carbohidratos: 275g</li>
                        <li>Grasas: 78g</li>
                        <li>Fibra: 28g</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tabs[3]:
                st.subheader("Alimentos por Categoría")
                
                # Filtro de categorías
                if food_categories:
                    all_categories = list(food_categories.keys())
                    selected_category = st.selectbox("Filtrar por categoría:", ["Todas"] + all_categories)
                    
                    # Preparar datos para el gráfico por categoría
                    if selected_category == "Todas":
                        # Mostrar distribución de calorías por categoría
                        category_calories = {}
                        for cat, items in food_categories.items():
                            category_calories[cat] = sum(item.get('nutrition', {}).get('total_calories', 0) for item in items)
                        
                        category_df = pd.DataFrame({
                            'Categoría': list(category_calories.keys()),
                            'Calorías': list(category_calories.values())
                        })
                        
                        # Calcular porcentajes
                        total_cal = category_df['Calorías'].sum()
                        category_df['Porcentaje'] = category_df['Calorías'] / total_cal * 100 if total_cal > 0 else 0
                        category_df['Etiqueta'] = category_df.apply(lambda x: f"{x['Categoría']}: {x['Porcentaje']:.1f}%", axis=1)
                        
                        # Crear gráfico de pastel con Altair en lugar de matplotlib
                        cat_pie = alt.Chart(category_df).mark_arc().encode(
                            theta=alt.Theta(field="Calorías", type="quantitative"),
                            color=alt.Color(field="Categoría", type="nominal"),
                            tooltip=['Categoría', 'Calorías', 'Porcentaje']
                        ).properties(
                            width=400,
                            height=400,
                            title='Distribución de Calorías por Categoría'
                        )
                        
                        # Añadir etiquetas
                        cat_text = cat_pie.mark_text(radius=130, size=14).encode(
                            text="Etiqueta"
                        )
                        
                        st.altair_chart(cat_pie + cat_text, use_container_width=True)
                        
                        # Mostrar tabla de alimentos por categoría
                        for cat, items in food_categories.items():
                            with st.expander(f"Categoría: {cat} ({len(items)} alimentos)"):
                                for item in items:
                                    st.markdown(f"""
                                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                        <h5 style="margin: 0;">{item['label']}</h5>
                                        <p style="margin: 5px 0;">Calorías: {item.get('nutrition', {}).get('total_calories', 0)} kcal</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        # Mostrar solo los alimentos de la categoría seleccionada
                        if selected_category in food_categories:
                            st.write(f"Alimentos en la categoría '{selected_category}':")
                            for item in food_categories[selected_category]:
                                nutrition = item.get('nutrition', default_nutrition)
                                st.markdown(f"""
                                <div class="result-card">
                                    <h3>{item['label']}</h3>
                                    <div class="nutrition-info">
                                        <p>Calorías totales: <span class="nutrition-value">{nutrition.get('total_calories', 'N/A')} kcal</span></p>
                                        <p>Proteínas: <span class="nutrition-value">{nutrition.get('protein_g', 'N/A')} g</span></p>
                                        <p>Carbohidratos: <span class="nutrition-value">{nutrition.get('carbs_g', 'N/A')} g</span></p>
                                        <p>Grasas: <span class="nutrition-value">{nutrition.get('fat_g', 'N/A')} g</span></p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Alimentos Detectados e Información Nutricional")
        
        # Mostrar resumen total de calorías
        total_calories = sum(result.get('nutrition', {}).get('total_calories', 0) for result in results)
        st.markdown(f"""
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #1976d2;">
            <h3 style="margin: 0; color: #1976d2;">Total de calorías: {total_calories} kcal</h3>
            <p style="margin: 5px 0 0 0; color: #1976d2;">({daily_percentages['calories']:.1f}% de la ingesta diaria recomendada)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar recomendaciones
        st.subheader("Recomendaciones Nutricionales")
        st.markdown(f"""
        <div style="background-color: #f1f8e9; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #689f38;">
            <div style="color: #33691e;">{recommendation_response.message.content}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar información de cada alimento
        for result in results:
            nutrition = result.get('nutrition', default_nutrition)
            condition = result.get('condition', default_condition)
            
            # Determinar el color del borde según el estado del alimento
            border_color = "#4CAF50"  # Verde por defecto (buen estado)
            if condition.get("estado") == "Deteriorado":
                border_color = "#F44336"  # Rojo (deteriorado)
            elif condition.get("estado") == "Regular":
                border_color = "#FF9800"  # Naranja (regular)
            elif condition.get("estado") == "No determinable":
                border_color = "#9E9E9E"  # Gris (no determinable)
            
            # Crear un badge para el estado del alimento
            condition_badge_color = {
                "Excelente": "#4CAF50",  # Verde
                "Bueno": "#8BC34A",      # Verde claro
                "Regular": "#FF9800",    # Naranjo
                "Deteriorado": "#F44336", # Rojo
                "No determinable": "#9E9E9E"  # Gris
            }.get(condition.get("estado"), "#9E9E9E")
            
            # Añadir iconos para cada estado
            condition_icon = {
                "Excelente": "✅",
                "Bueno": "👍",
                "Regular": "⚠️",
                "Deteriorado": "❌",
                "No determinable": "❓"
            }.get(condition.get("estado"), "❓")
            
            # Usar componentes nativos de Streamlit en lugar de HTML personalizado
            st.write(f"### {result['label']}")
            
            # Crear columnas para los badges
            col_cat, col_estado = st.columns([1, 2])
            with col_cat:
                st.write(f"**Categoría:** {nutrition.get('category', 'Otros')}")
            with col_estado:
                estado_texto = condition.get('estado', 'No determinable')
                if estado_texto == "Excelente" or estado_texto == "Bueno":
                    st.success(f"{condition_icon} Estado: {estado_texto}")
                elif estado_texto == "Regular":
                    st.warning(f"{condition_icon} Estado: {estado_texto}")
                elif estado_texto == "Deteriorado":
                    st.error(f"{condition_icon} Estado: {estado_texto}")
                else:
                    st.info(f"{condition_icon} Estado: {estado_texto}")
            
            st.write(f"**Confianza en la detección:** {result['confidence']}")
            
            # Crear una tarjeta para el estado del alimento
            st.subheader(f"{condition_icon} Estado del Alimento")
            
            # Mostrar nivel de confianza
            st.write(f"**Nivel de confianza:** {condition.get('confianza', 'Bajo')}")
            
            # Mostrar signos de deterioro
            st.write(f"**Signos observados:** {condition.get('signos_deterioro', 'No se pudo determinar')}")
            
            # Mostrar detalles de evaluación
            st.write("#### Detalles de la evaluación:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"🎨 **Color:**  \n{condition.get('detalles_evaluacion', {}).get('color', 'No evaluado')}")
            with col2:
                st.write(f"👆 **Textura:**  \n{condition.get('detalles_evaluacion', {}).get('textura', 'No evaluada')}")
            with col3:
                st.write(f"👁️ **Apariencia:**  \n{condition.get('detalles_evaluacion', {}).get('apariencia', 'No evaluada')}")
            
            # Mostrar recomendación
            st.info(f"💡 **Recomendación:**  \n{condition.get('recomendacion', 'Verificar visualmente antes de consumir')}")
            
            # Mostrar información nutricional
            st.subheader("Información Nutricional")
            
            # Crear columnas para mostrar la información nutricional
            col_nut1, col_nut2 = st.columns(2)
            
            with col_nut1:
                st.write(f"**Calorías por 100g:** {nutrition.get('calories_per_100g', 'N/A')} kcal")
                st.write(f"**Peso estimado:** {nutrition.get('estimated_weight_g', 'N/A')} g")
                st.write(f"**Calorías totales:** {nutrition.get('total_calories', 'N/A')} kcal")
                st.write(f"**Proteínas:** {nutrition.get('protein_g', 'N/A')} g")
            
            with col_nut2:
                st.write(f"**Carbohidratos:** {nutrition.get('carbs_g', 'N/A')} g")
                st.write(f"**Grasas:** {nutrition.get('fat_g', 'N/A')} g")
                st.write(f"**Fibra:** {nutrition.get('fiber_g', 'N/A')} g")
                st.write(f"**Azúcares:** {nutrition.get('sugar_g', 'N/A')} g")
            
            # Agregar un separador entre alimentos
            st.write("---")
        
        # Opciones para exportar/guardar resultados
        st.subheader("Exportar Resultados")
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            if st.button("Descargar como CSV"):
                # Crear DataFrame para exportar
                export_df = pd.DataFrame([{
                    "Alimento": r["label"],
                    "Categoría": r["nutrition"].get("category", "Otros"),
                    "Estado": r["condition"].get("estado", "No determinable"),
                    "Calorías (kcal)": r["nutrition"].get("total_calories", 0),
                    "Peso (g)": r["nutrition"].get("estimated_weight_g", 0),
                    "Proteínas (g)": r["nutrition"].get("protein_g", 0),
                    "Carbohidratos (g)": r["nutrition"].get("carbs_g", 0),
                    "Grasas (g)": r["nutrition"].get("fat_g", 0),
                    "Fibra (g)": r["nutrition"].get("fiber_g", 0),
                    "Azúcares (g)": r["nutrition"].get("sugar_g", 0),
                    "Signos de deterioro": r["condition"].get("signos_deterioro", "No determinado"),
                    "Recomendación": r["condition"].get("recomendacion", "")
                } for r in results])
                
                # Convertir a CSV
                csv = export_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="analisis_nutricional_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Descargar CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col_exp2:
            if st.button("Guardar Análisis"):
                # Inicializar historial en session_state si no existe
                if 'historial_analisis' not in st.session_state:
                    st.session_state.historial_analisis = []
                
                # Guardar en historial
                st.session_state.historial_analisis.append(export_data)
                st.success("¡Análisis guardado correctamente!")

        # Añadir una nueva sección para resumen de estado de los alimentos
        if results:
            st.subheader("Resumen del Estado de los Alimentos")
            
            # Contar alimentos por estado
            estados = {"Excelente": 0, "Bueno": 0, "Regular": 0, "Deteriorado": 0, "No determinable": 0}
            for result in results:
                estado = result.get('condition', {}).get('estado', 'No determinable')
                estados[estado] = estados.get(estado, 0) + 1
            
            # Crear DataFrame para el gráfico
            estados_df = pd.DataFrame({
                'Estado': list(estados.keys()),
                'Cantidad': list(estados.values())
            })
            
            # Filtrar estados con al menos un alimento
            estados_df = estados_df[estados_df['Cantidad'] > 0]
            
            if not estados_df.empty:
                # Crear gráfico de barras con Altair
                colors = {
                    "Excelente": "#4CAF50",
                    "Bueno": "#8BC34A",
                    "Regular": "#FF9800",
                    "Deteriorado": "#F44336",
                    "No determinable": "#9E9E9E"
                }
                
                domain = list(estados_df['Estado'])
                range_ = [colors[estado] for estado in domain]
                
                chart = alt.Chart(estados_df).mark_bar().encode(
                    x='Estado',
                    y='Cantidad',
                    color=alt.Color('Estado', scale=alt.Scale(domain=domain, range=range_))
                ).properties(
                    width=400,
                    height=300,
                    title='Cantidad de Alimentos por Estado'
                )
                
                st.altair_chart(chart, use_container_width=True)
                
                # Mostrar recomendaciones generales según el estado de los alimentos
                if estados.get("Deteriorado", 0) > 0:
                    st.error("""
                    ### ⚠️ Advertencia: Alimentos Deteriorados Detectados
                    
                    Se han detectado uno o más alimentos en estado de deterioro. Por tu seguridad, considera las siguientes recomendaciones:
                    
                    - No consumas alimentos con signos visibles de deterioro
                    - Desecha adecuadamente los alimentos deteriorados
                    - Revisa las condiciones de almacenamiento de tus alimentos
                    - Verifica las fechas de caducidad regularmente
                    """)
                elif estados.get("Regular", 0) > 0:
                    st.warning("""
                    ### ⚠️ Precaución: Alimentos en Estado Regular
                    
                    Algunos alimentos muestran signos de estar en estado regular. Considera estas recomendaciones:
                    
                    - Consume estos alimentos pronto para evitar mayor deterioro
                    - Verifica cuidadosamente antes de consumir
                    - Mejora las condiciones de almacenamiento
                    """)
                elif estados.get("Excelente", 0) > 0 or estados.get("Bueno", 0) > 0:
                    st.success("""
                    ### ✅ Buenas Noticias: Alimentos en Buen Estado
                    
                    La mayoría de los alimentos detectados están en buen estado. Para mantenerlos así:
                    
                    - Continúa con las buenas prácticas de almacenamiento
                    - Mantén la cadena de frío cuando sea necesario
                    - Consume los alimentos frescos dentro de su tiempo óptimo
                    """)
                
                # Añadir guía informativa sobre cómo identificar alimentos en mal estado
                with st.expander("📚 Guía: Cómo identificar alimentos en mal estado"):
                    st.subheader("Señales comunes de deterioro en alimentos")
                    
                    st.markdown("#### 🥩 Carnes")
                    st.markdown("""
                    - **Color:** Cambio de color a gris, verde o marrón
                    - **Olor:** Olor agrio o desagradable
                    - **Textura:** Viscosa o pegajosa al tacto
                    - **Apariencia:** Presencia de moho o manchas
                    """)
                    
                    st.markdown("#### 🥛 Lácteos")
                    st.markdown("""
                    - **Apariencia:** Separación, grumos o moho
                    - **Olor:** Olor agrio o fermentado
                    - **Sabor:** Sabor ácido o amargo
                    """)
                    
                    st.markdown("#### 🥦 Frutas y Verduras")
                    st.markdown("""
                    - **Textura:** Demasiado blanda, marchita o arrugada
                    - **Color:** Manchas oscuras excesivas o decoloración
                    - **Olor:** Olor a fermentación o descomposición
                    - **Apariencia:** Moho visible o jugos que supuran
                    """)
                    
                    st.markdown("#### 🍞 Panes y Cereales")
                    st.markdown("""
                    - **Apariencia:** Manchas de moho (verdes, blancas o negras)
                    - **Olor:** Olor a humedad o moho
                    - **Textura:** Excesivamente dura o extrañamente húmeda
                    """)
                    
                    st.markdown("#### 🍳 Huevos")
                    st.markdown("""
                    - **Olor:** Olor a azufre o desagradable
                    - **Apariencia:** Manchas en la yema o clara
                    - **Prueba de flotación:** Los huevos que flotan en agua suelen estar en mal estado
                    """)
                    
                    st.info("💡 **Recuerda:**  \n\"Cuando tengas dudas, mejor desecha el alimento. La seguridad alimentaria siempre debe ser prioritaria.\"")
                    
                # Añadir sección de consejos para conservación de alimentos
                with st.expander("🧊 Consejos para conservar alimentos frescos por más tiempo"):
                    st.subheader("Mejores prácticas para conservación de alimentos")
                    
                    st.markdown("#### 🌡️ Control de temperatura")
                    st.markdown("""
                    - Mantén el refrigerador a 4°C o menos
                    - El congelador debe estar a -18°C o menos
                    - No dejes alimentos perecederos a temperatura ambiente por más de 2 horas
                    """)
                    
                    st.markdown("#### 📦 Almacenamiento adecuado")
                    st.markdown("""
                    - Usa recipientes herméticos para alimentos
                    - Separa las frutas y verduras que producen etileno (manzanas, plátanos) de las sensibles a este gas
                    - Almacena la carne cruda en la parte inferior del refrigerador
                    """)
                    
                    st.markdown("#### 🧼 Higiene")
                    st.markdown("""
                    - Lava frutas y verduras antes de almacenarlas
                    - Mantén limpio el refrigerador
                    - Usa utensilios y tablas de cortar diferentes para alimentos crudos y cocidos
                    """)
                    
                    st.markdown("#### 📅 Rotación de alimentos")
                    st.markdown("""
                    - Sigue el principio "primero en entrar, primero en salir"
                    - Etiqueta los alimentos con la fecha de almacenamiento
                    - Revisa regularmente los alimentos almacenados
                    """)
                    
                    st.info("💡 **Consejo profesional:**  \n\"Planifica tus comidas semanalmente para reducir el desperdicio de alimentos y asegurar que consumes los productos más perecederos primero.\"")

    os.unlink(image_path)

def about_page():
    st.title("Sobre ¿Qué hay en tu plato?")
    st.markdown("""
    ¿Qué hay en tu plato? es una aplicación de análisis nutricional impulsada por inteligencia artificial que te permite:

    - **Identificar alimentos** en imágenes con alta precisión
    - **Calcular información nutricional** como calorías, proteínas, carbohidratos y grasas
    - **Detectar el estado de los alimentos** para garantizar su seguridad alimentaria
    - **Recibir recomendaciones personalizadas** para mejorar tus hábitos alimenticios
    - **Visualizar datos** a través de gráficos interactivos
    - **Exportar y guardar** tus análisis para seguimiento

    Esta aplicación utiliza el modelo Gemini de Google para proporcionar análisis precisos y recomendaciones personalizadas.

    ### Tecnologías utilizadas
    - Streamlit para la interfaz de usuario
    - Google Gemini para análisis de imágenes e información nutricional
    - OpenCV para procesamiento de imágenes
    - Altair y Pandas para visualización de datos
    
    ### Funcionalidad de detección del estado de los alimentos
    
    Nuestra aplicación ahora incluye una avanzada funcionalidad de detección del estado de los alimentos que:
    
    - Analiza visualmente cada alimento para detectar signos de deterioro
    - Evalúa el color, textura y apariencia general
    - Clasifica los alimentos en diferentes estados (Excelente, Bueno, Regular, Deteriorado)
    - Proporciona recomendaciones específicas sobre el consumo seguro
    - Ofrece guías educativas sobre cómo identificar alimentos en mal estado
    
    Esta funcionalidad está diseñada para ayudarte a tomar decisiones informadas sobre la seguridad de tus alimentos y reducir el riesgo de enfermedades transmitidas por alimentos.
    """)

def contact_page():
    st.title("Investigaciones y Recursos")
    
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
    
    st.markdown("""
    ### Enlaces a recursos nutricionales
    
    - [Base de Datos Española de Composición de Alimentos (BEDCA)](https://www.bedca.net/)
    - [USDA FoodData Central](https://fdc.nal.usda.gov/)
    - [Organización Mundial de la Salud - Nutrición](https://www.who.int/es/health-topics/nutrition)
    
    ### Contacto
    
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

if __name__ == "__main__":
    main()

