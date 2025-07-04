import pandas as pd
import json
import time
from mistralai import Mistral

# ✅ Set your Mistral API key
client = Mistral(api_key="8WLb8zitbnp5QSOSOnuT04hABhifT4u5")

# ✅ Load your CSV (must have 'description' column)
df = pd.read_csv("hm_descriptions.csv")

# ✅ Prompt Template
def build_prompt(description):
    return f"""
You are an expert product data annotator. Extract the following attributes from the product description and return them as a JSON object with these keys:

- Product Type
- Fabric/Material
- Fit
- Sleeve Type
- Support/Wiring
- Closures
- Neckline
- Waist Style
- Design Features
- Intended Use / Function
- Length

Only include values that are mentioned or strongly implied. If an attribute is missing, use "unknown".

Description:
\"\"\"{description}\"\"\"

Only return valid JSON. Do not include explanations or comments.
"""

# ✅ Entity extraction with retry & fallback model
def extract_structured_attributes(description):
    prompt = build_prompt(description)
    models_to_try = ["open-mistral-7b"]

    for model in models_to_try:
        for attempt in range(3):
            try:
                response = client.chat.complete(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                output = response.choices[0].message.content.strip()
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    return {"raw_output": output}
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️ Attempt {attempt+1} failed on model {model}: {error_msg}")
                if "429" in error_msg:
                    time.sleep(5)  # wait and retry if rate limited
                elif "401" in error_msg:
                    return {"error": "Unauthorized. Check your API key."}
                else:
                    return {"error": f"API error occurred: {error_msg}"}
    return {"error": "All model attempts failed."}

# ✅ Apply to all rows
df["structured_entities"] = df["description"].apply(extract_structured_attributes)

# ✅ Save to output file
df.to_csv("hm_structured_output.csv", index=False)
print("✅ Entity extraction complete. Output saved to 'hm_structured_output.csv'")
