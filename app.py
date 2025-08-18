from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import streamlit as st



# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="Phishing Email Generator", page_icon="🦙", layout="centered")
st.title("🦙 Phishing Email Generator for Educational Purposes with Llama3.2")
st.caption("Pick your constraints; the app formats the prompt and queries your local model.")

DEFAULT_TONES = [
    "neutral", "exciting", "formal", "urgent", "professional", "threatening", "authoritarion"
]


def build_prompt(length="short", tone="Neutral", scenario=None, extra=None):
    
    intro = "Provide phishing email for education purposes. "
    length_part = "Keep the email {}. ".format(length)
    tone_part = "Use {} tone. ".format(tone)
    
    prompt = intro + length_part + tone_part
    
    if scenario is not None:
        scenerio_part = "The email should follow a {} scenerio. ".format(scenario)
        prompt = prompt + scenerio_part

    if extra is not None:
        prompt = prompt + extra
    
    return prompt


def load_model_and_tokenizer(base_model_path, adapter_path, device_choice):

    use_cuda = (device_choice == "cuda" and torch.cuda.is_available())
    
    if not torch.cuda.is_available() and device_choice == "cuda":
        #print("cuda device is not available in the device. Please choose 'cpu' as Device")
        st.write("CUDA device is not available in local. Device is set as CPU by default.")
    
    dtype = torch.float16 if use_cuda else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, 
                                                      torch_dtype = dtype, 
                                                      device_map = "auto" if use_cuda else None)
                                                      #device_map = device_choice)
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return model, tokenizer, use_cuda



with st.sidebar:
    st.subheader("Model settings")
    base_model_path = st.text_input("Base model path / repo", value="meta-llama/Llama-3.2-1B-Instruct")
    adapter_path = st.text_input("PEFT adapter path", value="/Users/desidero/Desktop/phishing_chatbot")

    device_choice = st.selectbox("Device", options=["cpu", "cuda"], index=0,
                                 help="Choose 'cuda' if you have a GPU available.")
    
    st.divider()
    st.subheader("Generation")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top‑p", 0.1, 1.0, 0.95, 0.01)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.05, 0.01)
    seed_opt = st.toggle("Set random seed", value=False)
    seed_val = st.number_input("Seed", min_value=0, value=42, step=1, disabled=not seed_opt)


st.markdown("### Constraints")


# Length preset -> (min_words, max_words)
LENGTH_PRESETS = {
    "Short (≤ 100 words)": (1, 100),
    "Medium (100–200 words)": (100, 200),
    "Long (200–400 words)": (200, 400),
    "Custom …": None,
}


length = st.selectbox("Length of the email", options=list(LENGTH_PRESETS.keys()), placeholder="e.g., short, long, medium length, between 100 and 200 words ...")
tone = st.selectbox("Tone", options=DEFAULT_TONES, index=0)
scenario = st.text_input("Scenario (topic/goal), Optional", placeholder="e.g., product launch email for a new fitness app")
extra = st.text_area("Extra instructions, Optional", placeholder="e.g., include a CTA; avoid jargon")


if length == "Custom…":
    len_min, len_max = st.slider(
        "Custom word range",
        min_value=10, max_value=2000, value=(100, 200), step=10
    )
else:
    len_min, len_max = LENGTH_PRESETS[length]


prompt = build_prompt(length=length, tone=tone, scenario=scenario, extra=extra)

st.markdown("### Prompt Preview")
st.code(prompt, language="markdown")

generate = st.button("Generate")


if generate:
    with st.spinner("Loading model (first time may take a bit)…"):
        try:
            model, tokenizer, use_cuda = load_model_and_tokenizer(base_model_path, adapter_path, device_choice)

        except Exception as e:
            st.exception(e)
            st.stop()

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt")
    if use_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    try:
        with torch.no_grad():
            
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=len_max*1.3,
                temperature=float(temperature),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # If your base prompt appears in the output (common with chatty models), strip it
        generated_only = generated_text.replace(prompt, "").strip()

        generated = generated_only if generated_only else generated_text
                
        st.markdown("### Generated Text")
        st.write(generated)

        with st.expander("Details"):
            st.write({
                #"max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "device": "cuda" if use_cuda else "cpu"
            })

    except Exception as e:
        st.exception(e)

