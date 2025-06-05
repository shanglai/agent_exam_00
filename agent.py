import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent,LlmAgent
from google.genai import types
import pandas as pd
import json

path='C:/Users/David/Documents/projects/agents/agent_exam/agent_env/base.csv'

def get_data() -> dict:
    df= pd.read_csv(path)
    df= df.drop('n',axis=1)
    df.head()
    out_json= json.dumps(df.to_json(orient='records'))
    return {'status':200,'base_data':out_json}



question_ai= LlmAgent(
    name='question_ai',
    model='gemini-2.0-flash',
    description=(
        'Agente experto en experiencia de usuario en restaurantes que recibe una petición, carga datos base y busca cómo resolver la petición con los datos de base_data.'
    ),
    instruction=(
        """Eres un agente experto en experiencia de usuario en restaurantes que:
        * Recibe una petición
        * Usa la herramienta [get_data] para cargar los datos base
        * Busca cómo resolver la petición con los datos de base_data.
        * Si la petición no se puede responder porque está fuera del contexto de reseñas o comentarios relativo a la comida, servicios, atención o demás rubro de restaurantes, informa que no la puedes responder por no tener información al respecto.
        * Si se puede responder, plantéala de una forma clara. Ejemplo:
        ** Si requiere explicación de alguna pregunta o característica, simplemente responde en uno o dos párrafos, como:
        *** Pregunta: ¿Cuál es la queja más común en la sección 1? => Respuesta: La falta de limpieza.
        ** Si la respuesta tiene categorías, preséntala en forma de lista, como:  
        *** Pregunta: ¿Cuántas opiniones hay por restaurante (sección)? => Respuesta en forma de tabla: | Sección | Opiniones |    | 1 | 124 |, etc...."
        """
    ),
    tools=[get_data],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.4,
        max_output_tokens=8080
    )
)


root_agent = Agent(
    name="root_agent",
    model="gemini-2.0-flash",
    description=(
        "Agente experto en experiencia de usuario en restaurantes que da la bienvenida, carga la data con la función [get_data]" # y transfiere al agente <question_ai>."
    ),
    instruction=(
        """Eres un agente experto en experiencia de usuario en restaurantes que:
        * Da la bienvenida
        * Valida que la petición del usuario se pueda entender y si no se entiende, pedir que clarifique su respuesta.
        *** Ejemplo 1: Si el usuario escribe sin sentidos como "asdasd", solicita que reformule su pregunta.
        *** Ejemplo 2: Si el usuario escribe algo fuera del tema, indica que no puede responder esa pregunta.
        * Si se entiende, entonces indicar la petición redactada con orden y lógica
        * Envía la pregunta a <question_ai>.
        """
    ),
    sub_agents=[question_ai]
)