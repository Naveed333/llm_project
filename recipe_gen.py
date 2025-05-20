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
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("Please set your HF token in $HUGGINGFACE_TOKEN")
token_args = {"token": hf_token}

# ——— Device & (optional) 8-bit config ———
device = "cuda" if torch.cuda.is_available() else "cpu"
bnb_config = BitsAndBytesConfig(load_in_8bit=True)


def load_gemma(model_name: str = "google/gemma-3-1b-it"):
    # 1) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
        **token_args,
    )

    # 2) model: only quantize on GPU
    if torch.cuda.is_available():
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                **token_args,
            )
            print(f"✅ Loaded 8-bit quantized {model_name}")
        except Exception as e:
            print(f"⚠️ 8-bit quant failed ({e}); loading FP32 instead.")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                **token_args,
            )
    else:
        print("⚠️ No CUDA GPU detected — loading full-precision on CPU.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **token_args,
        )

    model.to(device)
    model.eval()
    return tokenizer, model


# Load Gemma once
tokenizer, model = load_gemma()
print(f"Using device: {device}  |  Model device: {model.device}")


def generate_text(
    prompt: str,
    max_new_tokens: int = 750,
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
    return text.split("[/INST]")[-1].strip()


def generate_recipe(
    ingredients_list: str,
    cuisine: str,
    difficulty: str,
    meal: str,
    preferences: str,
    recipe_name: str,
    temperature: float = 0.7,
) -> str:
    prompt = (
        "<s>[INST]\n"
        f"Create a detailed recipe for: {recipe_name} but using these ingredients: {ingredients_list}\n"
        "If the user provides an incorrect or faulty recipe name or ingredients, do not hallucinate—"
        "instead generate a random recipe using the given ingredients and include at the end:\n"
        "“⚠️ DISCLAIMER: The user-supplied recipe name or ingredients were invalid; this recipe is a generated approximation.”\n\n"
        f"Cuisine type: {cuisine}\n"
        f"Cook Time: {difficulty}\n"
        f"Meal type: {meal}\n"
        f"Additional preferences: {preferences}\n\n"
        "If any of the ingredients include pork, bacon, ham, lard, or other non-halal items, include at the end of the recipe:\n"
        "“⚠️ DISCLAIMER: This recipe contains non-halal ingredients and is intended for consumers who do not follow halal dietary restrictions.”\n\n"
        "The recipe should include:\n"
        "1. Title\n"
        "2. Serving\n"
        "3. Ingredients list with quantities\n"
        "4. Step-by-step instructions\n"
        "5. Prep and cook times\n"
        "6. Nutritional info per serving (calories, macros)\n"
        "7. Health Data: for each serving, give % of daily recommended values, dietary tags (e.g., vegan, keto, gluten-free), and note key health benefits or cautions\n"
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
