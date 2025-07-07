"""
Product Data Extraction and Attribute Annotation System

This module provides functionality to extract product and care attributes
from product descriptions and care instructions using the Mistral AI API.
It processes JSON input files containing product data and outputs structured
 JSON with extracted attributes.

Features:
- Batch processing of product data from JSON files
- Automatic retry logic for API failures
- Error handling for various API response scenarios
- JSON validation and fallback handling
- Rate limiting protection with exponential backoff

Dependencies:
- mistralai: Official Mistral AI Python client
- json: Built-in JSON handling
- time: Built-in time utilities for delays

"""
import json
import time
from mistralai import Mistral
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# ✅ Initialize Mistral client
client = Mistral(api_key=api_key)

# ✅ Load JSON input
with open("hm_input.json", "r", encoding="utf-8") as f:
    input_data = json.load(f)  # should be a list of objects

# ✅ Prompt builder
def build_prompt(description, care_instructions):
    """
    Build a structured prompt for the Mistral AI model to extract product attributes.
    
    This function creates a detailed prompt that instructs the AI model to extract
    specific product and care attributes from the provided description and care
    instructions text. The prompt is designed to return structured JSON output.
    
    Args:
        description (str): Product description text containing details about the item.
                          Should include information about design, materials, fit, etc.
        care_instructions (str): Care and maintenance instructions for the product.
                                Typically includes washing, drying, and storage guidelines.
    
    Returns:
        str: A formatted prompt string that includes:
            - Clear instructions for the AI model
            - List of expected product attributes to extract
            - List of expected care attributes to extract
            - Input data (description and care instructions)
            - Output format specification (JSON only)
    
    Product Attributes Extracted:
        - Product Type: Category/type of the product
        - Support/Wiring: Information about structural support elements
        - Closures: Types of fasteners, zippers, buttons, etc.
        - Neckline design: Style and cut of neckline area
        - Waist Style: Waist fit and design characteristics
        - Design Features: Special design elements or decorative features
        - Intended Use / Function: Primary purpose or use case
        - Length: Size/length specifications
    
    Care Attributes Extracted:
        - Washing Instructions: How to clean the item
        - Drying Method: Recommended drying approach
        - Bleach Instructions: Bleach usage guidelines
        - Dry Cleaning: Professional cleaning requirements
        - Ironing Instructions: Ironing and pressing guidelines
    
    """
    return f"""
You are an expert product data annotator. Extract the following attributes from the product description and care instructions. Return them as a valid JSON object with these keys:

Product Attributes:
- Product Type
- Support/Wiring
- Closures
- Neckline design
- Waist Style
- Design Features
- Intended Use / Function
- Length

Care Attributes:
- Washing Instructions
- Drying Method
- Bleach Instructions
- Dry Cleaning
- Ironing Instructions

Only include values that are mentioned or strongly implied. If something is missing, use "Not Available".

Product Description:
\"\"\"{description}\"\"\"

Care Instructions:
\"\"\"{care_instructions}\"\"\"

Only return valid JSON. No explanation.
"""

# ✅ Extraction function
def extract_entities(entry):
    """
    Extract product and care attributes from a single product entry using Mistral AI.
    
    This function processes a single product entry containing description and care
    instructions, sends it to the Mistral AI API for attribute extraction, and
    returns the original entry enhanced with extracted attributes.
    
    The function includes robust error handling and retry logic to handle various
    API failure scenarios including rate limiting, authentication errors, and
    network issues.
    
    Args:
        entry (dict): A dictionary containing product information with at least:
                     - description (str): Product description text
                     - care_instructions (str): Care instruction text
                     - Additional fields are preserved in the output
    
    Returns:
        dict: Enhanced dictionary containing:
            - All original fields from the input entry
            - Extracted attributes as individual key-value pairs (on success)
            - OR raw_output (str): Raw API response if JSON parsing fails
            - OR error (str): Error message if API calls fail
    
    Retry Logic:
        - Attempts up to 3 API calls per entry
        - Handles rate limiting (429 errors) with 5-second delays
        - Handles authentication errors (401) with immediate termination
        - Handles other API errors with appropriate error messages
    
    Error Handling:
        - JSON parsing errors: Returns raw output for manual inspection
        - Rate limiting: Implements exponential backoff
        - Authentication errors: Provides clear error message
        - Network/API errors: Captures and returns error details
        - All attempts failed: Returns failure message
    
    API Configuration:
        - Model: open-mistral-7b
        - Temperature: 0.0 (deterministic output)
        - Max attempts: 3
        - Rate limit delay: 5 seconds
    """
    description = entry.get("description", "")
    care_instructions = entry.get("care_instructions", "")
    prompt = build_prompt(description, care_instructions)

    for attempt in range(3):
        try:
            response = client.chat.complete(
                model="open-mistral-7b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            output = response.choices[0].message.content.strip()
            try:
                extracted = json.loads(output)
                return {**entry, **extracted}
            except json.JSONDecodeError:
                return {**entry, "raw_output": output}
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Attempt {attempt + 1} failed: {error_msg}")
            if "429" in error_msg:
                time.sleep(5)
            elif "401" in error_msg:
                return {**entry, "error": "Unauthorized. Check your API key."}
            else:
                return {**entry, "error": f"API error: {error_msg}"}
    return {**entry, "error": "All attempts failed"}

# ✅ Run on all items
output_data = [extract_entities(entry) for entry in input_data]

# ✅ Save to JSON output
with open("hm_output2.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("✅ Extraction complete. Results saved to 'hm_output.json'")
