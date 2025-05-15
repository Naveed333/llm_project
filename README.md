# IngrEdibles

A Streamlit-powered web app that lets you snap or upload a photo of vegetables (or type them in), detects the ingredients via CLIP, and generates full recipes using a quantized LLM from Hugging Face. Free users get a general recipe; paid subscribers unlock personalized recipes (spice level, cuisine, cook time, dietary goals).

---

## 🔍 Features

- **Vegetable Detection**  
  Uses OpenAI’s CLIP via `detect.py` to identify all veggies in an uploaded image.
- **User Accounts & Subscriptions**  
  Register & log in, upgrade to a Paid plan, cancel subscription, and log out.
- **Preferences**  
  Save sliders & selectors for spice level, serving size, cuisine, meal type, cook time, and health goals.
- **Recipe Generation**  
  Calls a quantized LLM (primary: `google/gemma-3-1b-it`, fallback: `tiiuae/falcon-7b-instruct`) via `recipe_gen.py` to produce detailed recipes.
- **Dynamic UI**  
  Modern, tab-based layout:  
  1. **Home** – detect ingredients & generate recipes  
  2. **Preferences** – view/edit your saved settings  
  3. **Profile** – manage account, subscription, and logout  

---


### Setting Up the Environment

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/Naveed333/llm_project
    cd llm_project
    ```

2. **Create and Activate a Virtual Environment**:

    ```bash
    python3 -m venv venv  # Create virtual environment
    source venv/bin/activate  # Activate it (for macOS/Linux)
    # or venv\Scripts\activate for Windows
    ```

3. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

 4. **Run Streamlit for Real-Time Visualization**:

    Start the Streamlit app to visualize the data in real-time.

    ```bash
    streamlit run app.py
    ```