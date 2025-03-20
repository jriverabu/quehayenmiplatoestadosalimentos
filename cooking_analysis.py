# cooking_analysis.py
import streamlit as st
import re
import json
from PIL import Image
from google.generativeai.types import ChatMessage, MessageRole, TextBlock, ImageBlock

def analyze_cooking_status(temp_filename, gemini_pro):
    """
    Analiza el estado de cocción de alimentos en una imagen.
    
    Args:
        temp_filename: Ruta al archivo de imagen temporal
        gemini_pro: Instancia del modelo Gemini Pro
    
    Returns:
        None (muestra resultados directamente en la interfaz de Streamlit)
    """
    st.subheader("Análisis del Estado del Alimento")
    
    # Añadir opción para detectar si el alimento está crudo
    cooking_status = st.checkbox("Detectar nivel de cocción del alimento", value=True,
                               help="Determina si el alimento está crudo, parcialmente cocinado o totalmente cocinado")
    
    with st.spinner("Analizando estado del alimento..."):
        st.info("Procesando imagen para evaluar el estado y calidad del alimento...")
        
        # Implementar análisis real del estado con Gemini
        try:
            # Modificar el mensaje para incluir análisis de cocción si está activado
            prompt_text = """Analiza esta imagen de comida y evalúa el estado y calidad de cada alimento visible.
            Para cada alimento:
            1. Identifica su nombre
            2. Evalúa su estado (Excelente, Bueno, Regular o Deteriorado)
            3. Describe brevemente los detalles visuales que indican su estado
            4. Proporciona recomendaciones sobre su consumo
            """
            
            # Añadir instrucciones para detectar nivel de cocción si está activado
            if cooking_status:
                prompt_text += """
            5. Determina el nivel de cocción (Crudo, Parcialmente cocinado, Completamente cocinado)
            6. Indica si es seguro consumirlo en su nivel actual de cocción
            """
            
            prompt_text += """
            Responde SOLO con un objeto JSON con el siguiente formato (sin texto adicional):
            [
              {
                "alimento": "nombre_del_alimento",
                "estado": "Excelente/Bueno/Regular/Deteriorado",
                "detalles": "descripción_detallada_visual",
                "confianza": valor_entre_0_y_1,
                "recomendaciones": "recomendación_sobre_consumo"
            """
            
            # Añadir campos adicionales para el nivel de cocción
            if cooking_status:
                prompt_text += """,
                "nivel_coccion": "Crudo/Parcialmente cocinado/Completamente cocinado",
                "seguro_consumo": true/false,
                "tiempo_coccion_recomendado": "tiempo adicional recomendado (solo si aplica)"
            """
            
            prompt_text += """
              },
              ...
            ]"""
            
            # Crear mensaje para Gemini
            food_condition_msg = ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    TextBlock(text=prompt_text),
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
                            cooking_status_html = ""
                            if cooking_status and "nivel_coccion" in item:
                                # Determinar color para nivel de cocción
                                if item["nivel_coccion"] == "Crudo":
                                    cooking_color = "#F44336"  # Rojo
                                    cooking_icon = "🥩"
                                elif item["nivel_coccion"] == "Parcialmente cocinado":
                                    cooking_color = "#FF9800"  # Naranja
                                    cooking_icon = "🔥"
                                else:  # Completamente cocinado
                                    cooking_color = "#4CAF50"  # Verde
                                    cooking_icon = "👨‍🍳"
                                
                                # Mostrar seguridad de consumo
                                safe_text = "Seguro para consumo" if item.get("seguro_consumo", False) else "No seguro para consumo"
                                safe_color = "#4CAF50" if item.get("seguro_consumo", False) else "#F44336"
                                
                                cooking_status_html = f"""
                                <div style="margin-top: 15px; padding: 12px; border-radius: 8px; background-color: #f8f9fa; border: 1px solid {cooking_color};">
                                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                        <span style="font-size: 1.5em; margin-right: 8px;">{cooking_icon}</span>
                                        <span style="font-weight: 600; color: {cooking_color};">{item["nivel_coccion"]}</span>
                                    </div>
                                    <div style="margin-top: 5px; padding: 5px 10px; background-color: {safe_color}; color: white; border-radius: 20px; text-align: center; font-size: 0.85em;">
                                        {safe_text}
                                    </div>
                                </div>
                                """
                                
                                if "tiempo_coccion_recomendado" in item and item["nivel_coccion"] != "Completamente cocinado":
                                    cooking_status_html += f"""
                                    <div style="margin-top: 10px; padding: 8px; background-color: #fff8e1; border-radius: 5px; font-size: 0.9em;">
                                        <strong>Tiempo adicional:</strong> {item["tiempo_coccion_recomendado"]}
                                    </div>
                                    """
                            
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
                                {cooking_status_html}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Añadir botón para instrucciones específicas
                            if st.button(f"📋 Ver guía para {item['alimento']}", key=f"guide_{item['alimento']}"):
                                st.info(f"Mostrando información detallada para {item['alimento']}...")
                        
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
                            
                            with condition_tabs[1]:
                                st.markdown(f"""
                                <h4 style="color: #2c3e50; border-bottom: 2px solid {color}; padding-bottom: 8px;">Análisis Detallado</h4>
                                """, unsafe_allow_html=True)
                                
                                # Mostrar información de análisis basada en el estado
                                if cooking_status and "nivel_coccion" in item:
                                    cooking_level = item["nivel_coccion"]
                                    
                                    if cooking_level == "Crudo":
                                        st.warning("⚠️ Este alimento está crudo y puede requerir cocción para consumo seguro.")
                                        
                                        # Información específica basada en el tipo de alimento
                                        food_type = item['alimento'].lower()
                                        if any(meat in food_type for meat in ["pollo", "pavo", "ave"]):
                                            st.markdown("""
                                            **Recomendaciones para aves:**
                                            - Cocinar hasta que la temperatura interna alcance 74°C (165°F)
                                            - Verificar que no haya partes rosadas en el centro
                                            - Evitar la contaminación cruzada limpiando superficies y utensilios
                                            """)
                                        elif any(meat in food_type for meat in ["res", "ternera", "steak", "filete"]):
                                            st.markdown("""
                                            **Recomendaciones para carne de res:**
                                            - Temperatura mínima recomendada: 63°C (145°F) con 3 minutos de reposo
                                            - Niveles de cocción:
                                              - Poco hecho: 55-60°C (130-140°F)
                                              - Término medio: 60-65°C (140-150°F)
                                              - Bien cocido: +71°C (+160°F)
                                            """)
                                        elif any(fish in food_type for fish in ["pescado", "atún", "salmón", "pez"]):
                                            st.markdown("""
                                            **Recomendaciones para pescado:**
                                            - Cocinar hasta 63°C (145°F) o hasta que la carne esté opaca y se separe fácilmente
                                            - Si desea consumirlo crudo, asegúrese de que sea apto para consumo crudo y haya sido previamente congelado para eliminar parásitos
                                            """)
                                        else:
                                            st.markdown("""
                                            **Recomendaciones generales:**
                                            - Cocinar completamente antes de consumir
                                            - Seguir las instrucciones específicas para este tipo de alimento
                                            """)
                                    
                                    elif cooking_level == "Parcialmente cocinado":
                                        st.warning("⚠️ Este alimento está parcialmente cocinado y puede requerir cocción adicional.")
                                        
                                        if "tiempo_coccion_recomendado" in item:
                                            st.info(f"**Tiempo de cocción adicional recomendado**: {item['tiempo_coccion_recomendado']}")
                                        
                                        st.markdown("""
                                        **Pasos recomendados:**
                                        1. Continuar la cocción hasta alcanzar la temperatura interna segura
                                        2. Verificar el centro del alimento para asegurarse de que esté completamente cocinado
                                        3. Utilizar un termómetro de cocina para mayor precisión
                                        """)
                                    
                                    else:  # Completamente cocinado
                                        st.success("✅ Este alimento está completamente cocinado y listo para consumir.")
                                        
                                        st.markdown("""
                                        **Recomendaciones:**
                                        - Mantener caliente (por encima de 60°C/140°F) si no se va a consumir inmediatamente
                                        - Refrigerar dentro de las 2 horas de cocción si se va a almacenar
                                        - Consumir refrigerado dentro de los siguientes 3-4 días
                                        """)
                            
                            with condition_tabs[2]:
                                st.markdown(f"<h4 style='color: #2c3e50; border-bottom: 2px solid {color}; padding-bottom: 8px;'>Recomendaciones</h4>", unsafe_allow_html=True)
                                st.write(item['recomendaciones'])
                                
                                # Añadir recomendaciones adicionales basadas en el estado
                                st.markdown("#### Acciones recomendadas")
                                
                                actions = []
                                if item["estado"] == "Excelente":
                                    actions = [
                                        "✅ Consumir con confianza",
                                        "✅ Almacenar adecuadamente para mantener frescura",
                                        "✅ Ideal para todas las preparaciones"
                                    ]
                                elif item["estado"] == "Bueno":
                                    actions = [
                                        "✅ Apto para consumo",
                                        "✅ Consumir preferentemente pronto",
                                        "⚠️ Verificar que mantenga buen aspecto durante la preparación"
                                    ]
                                elif item["estado"] == "Regular":
                                    actions = [
                                        "⚠️ Consumir con precaución",
                                        "⚠️ Cocinar completamente antes de consumir",
                                        "⚠️ Desechar partes que muestren signos de deterioro",
                                        "⚠️ No recomendado para personas con sistema inmunológico comprometido"
                                    ]
                                elif item["estado"] == "Deteriorado":
                                    actions = [
                                        "❌ No recomendado para consumo",
                                        "❌ Desechar adecuadamente",
                                        "❌ Riesgo de intoxicación alimentaria"
                                    ]
                                
                                for action in actions:
                                    st.markdown(f"- {action}")
                        
                        # Línea divisoria entre elementos
                        st.markdown("<hr>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error al analizar el estado del alimento con IA: {str(e)}")