import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from project root (works regardless of current terminal cwd).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

def _get_api_key():
    # Support both env var names used by Gemini SDK/docs.
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

def _configure_genai():
    api_key = _get_api_key()
    if api_key:
        genai.configure(api_key=api_key)
    return api_key


def generate_insights(df):
    api_key = _configure_genai()

    if "Customer_Text" not in df.columns:
        df["insights"] = "Error: Customer_Text column is missing."
        return df

    if not api_key:
        df["insights"] = "Error: GOOGLE_API_KEY/GEMINI_API_KEY is not set in environment."
        return df

    insights = []

    prompt = """
You are a business analyst.
Analyze the following informations from customer profile and provide:

1. Key insights
2. Reason
3. Business Risk
4. Recommended Action
Keep it concise.
"""

    model_candidates = [
        "models/gemini-1.5-pro",
        "models/gemini-2.5-pro",
        "models/gemini-pro",
    ]

    for text in df["Customer_Text"]:
        last_error = None
        generated = None
        for model_name in model_candidates:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt + "\n\n" + str(text))
                generated = response.text
                break
            except Exception as e:
                last_error = str(e)
                if "API_KEY_INVALID" in last_error or "API key expired" in last_error:
                    generated = (
                        "Error: GOOGLE_API_KEY is invalid or expired. "
                        "Renew it in Google AI Studio and update it in .env (GOOGLE_API_KEY=...)."
                    )
                    break
                if "404" in last_error or "not found" in last_error.lower():
                    continue
                if "429" in last_error or "quota" in last_error.lower():
                    generated = (
                        "Key insights: Segment shows mixed engagement.\n"
                        "Reason: Based on average recency, order frequency, and order value.\n"
                        "Business Risk: Medium.\n"
                        "Recommended Action: Run targeted retention and upsell campaign.\n"
                        "Note: Gemini quota exceeded, showing fallback insight."
                    )
                    break
                break

        if generated is None:
            generated = f"Error: {last_error}" if last_error else "Error: Could not generate insights."
        insights.append(generated)

    df["insights"] = insights
    return df