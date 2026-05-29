
import pandas as pd
from dotenv import load_dotenv        
import os
import json
import re

from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from groq import Groq

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing. Set it as an environment variable")

# Configure Groq client
client = Groq(api_key=groq_api_key)

# Use Groq model (defaults to llama-3.1-8b-instant to avoid daily token rate limits)
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# -------------------------------
# State Definition
# -------------------------------
class CleaningState(BaseModel):
    input_text: str 
    structured_response: str = ""

# -------------------------------
# AI Agent
# -------------------------------
class AIAgent:
    def __init__(self):
        self.graph = self.create_graph()

    def create_graph(self):
        graph = StateGraph(CleaningState)

        def agent_logic(state: CleaningState) -> CleaningState:
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "user", "content": state.input_text}
                    ],
                    temperature=0.1,
                    max_tokens=2048,   # 100-row JSON needs ~1500-2000 tokens; 4096 was wasteful
                )
                response_text = response.choices[0].message.content
                return CleaningState(
                    input_text=state.input_text,
                    structured_response=response_text
                )
            except Exception as e:
                return CleaningState(
                    input_text=state.input_text,
                    structured_response=f"ERROR: {str(e)}"
                )
        
        graph.add_node("cleaning_agent", agent_logic)
        graph.add_edge("cleaning_agent", END)
        graph.set_entry_point("cleaning_agent")
        return graph.compile()
    
    def process_data(self, df: pd.DataFrame, batch_size: int = 30):
        """
        Process DataFrame in batches through the LLM.
        batch_size=30 is safe for max_tokens=2048 (30 rows × ~60 tokens/row ≈ 1800 tokens).
        """
        combined_records = []

        for i in range(0, len(df), batch_size):
            df_batch = df.iloc[i: i + batch_size]

            prompt = f"""You are an AI Data Cleaning Agent. Clean this dataset:

{df_batch.to_string()}

Tasks:
- Impute missing values (mean for numbers, mode for text)
- Remove duplicates
- Normalize numeric values
- Format text consistently

CRITICAL: Return ONLY a valid JSON array. No explanations, no markdown, no code blocks.
Output format: [{{"col1": "val1", "col2": 123}}, ...]"""

            state = CleaningState(input_text=prompt, structured_response="")
            response = self.graph.invoke(state)

            if isinstance(response, dict):
                response = CleaningState(**response)

            res_text = response.structured_response.strip()

            # If there's an error block returned by the graph node:
            if res_text.startswith("ERROR:"):
                return res_text

            # Extract JSON list using regex
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', res_text)
            if json_match:
                res_text = json_match.group(1).strip()

            array_match = re.search(r'\[[\s\S]*\]', res_text)
            if array_match:
                res_text = array_match.group(0)

            try:
                batch_records = json.loads(res_text)
                if isinstance(batch_records, list):
                    combined_records.extend(batch_records)
                else:
                    combined_records.append(batch_records)
            except Exception as e:
                # Fallback to traditional cleaning on formatting issues
                return f"ERROR: Failed to parse batch starting at row {i}: {str(e)}. Raw: {res_text}"

        return json.dumps(combined_records)
