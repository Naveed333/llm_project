# recipe_gen.py

import os
import torch
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from typing import List, Dict

# ——— Hugging Face authentication ———
# Read token from env var if you’ve set one via `export HUGGINGFACE_TOKEN=hf_xxx`
hf_token = os.getenv("HUGGINGFACE_TOKEN")
token_args = {"use_auth_token": hf_token} if hf_token else {}

# ——— Device & 8-bit config ———
device = "cuda" if torch.cuda.is_available() else "cpu"
bnb_config = BitsAndBytesConfig(load_in_8bit=True)


def load_model_and_tokenizer(model_name: str):
    """
    Attempt to load an 8-bit quantized model with device_map; on
    failure (missing accelerate or unsupported), fall back to full-precision.
    """
    # 1) tokenizer (always small)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **token_args)

    # 2) model: try quantized + device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", **token_args
        )
        print(f"✅ Loaded 8-bit quantized {model_name}")
    except (ValueError, ImportError) as e:
        # could be missing accelerate or unsupported quantization
        print(f"⚠️ 8-bit load failed for {model_name} ({e}); loading FP32 instead.")
        model = AutoModelForCausalLM.from_pretrained(model_name, **token_args)
        model.to(device)

    model.eval()
    return tokenizer, model


# ——— Primary & fallback ———
primary = "google/gemma-3-1b-it"
fallback = "tiiuae/falcon-7b-instruct"

try:
    tokenizer, model = load_model_and_tokenizer(primary)
except Exception as exc:
    print(
        f"⚠️ Could not load primary model {primary} ({exc}); falling back to {fallback}"
    )
    tokenizer, model = load_model_and_tokenizer(fallback)

print(f"Using device: {device}  |  Model device: {model.device}")


def generate_text(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("[/INST]")[-1].strip() if "[/INST]" in text else text


def generate_recipe(
    ingredients_list: str,
    cuisine: str,
    difficulty: str,
    meal: str,
    preferences: str,
    temperature: float = 0.7,
) -> str:
    prompt = (
        "<s>[INST]\n"
        f"Create a detailed recipe using these ingredients: {ingredients_list}.\n"
        f"Cuisine type: {cuisine}\n"
        f"Difficulty level: {difficulty}\n"
        f"Meal type: {meal}\n"
        f"Additional preferences: {preferences}\n\n"
        "The recipe should include:\n"
        "1. Title\n"
        "2. Ingredients list with quantities\n"
        "3. Step-by-step instructions\n"
        "4. Prep and cook times\n"
        "5. Nutritional info per serving (calories, macros)\n"
        "6. Health Data: for each serving, give % of daily recommended values, "
        "dietary tags (e.g., vegan, keto, gluten-free), and note key health benefits or cautions\n"
        "[/INST]"
    )
    return generate_text(prompt, temperature=temperature)


def get_default_questions() -> List[Dict]:
    return [
        {
            "question": "Do you prefer your dish spicy?",
            "options": ["Not spicy", "Mild spice", "Medium spice", "Very spicy"],
        },
        {
            "question": "Any dietary restrictions?",
            "options": ["None", "Vegetarian", "Vegan", "Gluten-free", "Low-carb"],
        },
        {
            "question": "Cooking method preference?",
            "options": ["Baking", "Grilling", "Stovetop", "Slow cooking", "Air frying"],
        },
        {
            "question": "Texture preference?",
            "options": ["Crispy", "Soft", "Crunchy", "Creamy", "Mixed textures"],
        },
        {
            "question": "Flavor profile preference?",
            "options": ["Savory", "Sweet", "Tangy", "Umami", "Balanced"],
        },
    ]


def generate_specific_question(ingredients_list: str) -> Dict:
    ings = [i.strip().lower() for i in ingredients_list.split(",")]
    if any(p in ings for p in ["chicken", "beef", "pork", "fish", "seafood", "meat"]):
        return {
            "question": "How do you like your meat cooked?",
            "options": ["Rare", "Medium", "Well done", "Extra well"],
        }
    if any(p in ings for p in ["pasta", "rice", "grains", "noodles"]):
        return {
            "question": "Pasta/grain doneness?",
            "options": ["Al dente", "Soft", "Very soft"],
        }
    if any(
        p in ings for p in ["broccoli", "carrot", "zucchini", "vegetable", "veggies"]
    ):
        return {
            "question": "Veggie texture?",
            "options": ["Crisp", "Tender-crisp", "Well-cooked", "Soft"],
        }
    return {
        "question": "Any garnish preference?",
        "options": ["Herbs", "Cheese", "Nuts/Seeds", "None"],
    }


def generate_recipe_options(
    ingredients: str, cuisine: str, difficulty: str, meal: str, preferences: str
) -> List[Dict[str, str]]:
    styles = ["simple and quick", "elaborate and impressive", "creative and unique"]
    temps = [0.7, 0.8, 0.9]
    results = []
    for style, temp in zip(styles, temps):
        full = generate_recipe(
            ingredients,
            cuisine,
            difficulty,
            meal,
            preferences + f"; for this option, make it {style}",
            temperature=temp,
        )
        title = next((ln for ln in full.splitlines() if ln.strip()), style.title())
        if len(title) > 60:
            title = title[:57] + "…"
        results.append({"title": title, "full_recipe": full})
    return results
