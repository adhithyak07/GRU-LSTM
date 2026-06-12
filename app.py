import streamlit as st
import numpy as np
import time

# --- Page Setup ---
st.set_page_config(
    page_title="GRU Simple Explainer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 GRU Step-by-Step Explainer")
st.write("See how a Gated Recurrent Unit (GRU) processes text token by token to understand sentiment.")
st.divider()

# --- Simulation Engine ---
def run_simple_gru(text):
    tokens = [t.strip() for t in text.split() if t.strip()]
    if not tokens:
        return None, 0.5, []
    
    # Simple 2-dimensional hidden state for easy viewing
    h_t = np.zeros(2)
    history = []
    
    for idx, token in enumerate(tokens):
        # Create a predictable pseudo-random state based on the word
        seed = sum(ord(c) for c in token) % 100
        np.random.seed(seed)
        
        # Gates determine what to keep and what to forget
        reset_gate = np.random.uniform(0.1, 0.9, 2)
        update_gate = np.random.uniform(0.1, 0.9, 2)
        
        # Sentiment score lookup
        clean_token = token.lower().strip(".,!?\"'")
        bias = 0.0
        if clean_token in ['fantastic', 'incredible', 'great', 'love', 'good']:
            bias = 0.6
        elif clean_token in ['bad', 'terrible', 'worst', 'boring', 'poor']:
            bias = -0.6
            
        # Math calculation
        candidate = np.tanh(reset_gate * h_t + np.random.normal(bias, 0.1, 2))
        h_t = (1 - update_gate) * h_t + update_gate * candidate
        h_t = np.clip(h_t, -1.0, 1.0)
        
        history.append({
            "step": idx + 1,
            "token": token,
            "reset": np.round(reset_gate, 2),
            "update": np.round(update_gate, 2),
            "state": np.round(h_t, 2)
        })
        
    final_score = 1 / (1 + np.exp(-np.mean(h_t) * 3))
    return final_score, history

# --- Layout Split ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Enter Your Text")
    user_input = st.text_input(
        "Type a sentence to analyze:",
        value="The story structure was great, but the pacing felt bad."
    )
    
    st.markdown("""
    ### How does a GRU work?
    * **Reset Gate:** Decides how much of the past memory to ignore.
    * **Update Gate:** Decides how much of the new information to keep.
    """)

with col_right:
    st.subheader("2. Final Result")
    
    if user_input.strip():
        score, logs = run_simple_gru(user_input)
        
        # Display simplified prediction metric
        if score >= 0.5:
            st.success(f"🟢 **Positive Sentiment** (Confidence: {score:.1%})")
        else:
            st.error(f"🔴 **Negative Sentiment** (Confidence: {(1 - score):.1%})")
    else:
        st.info("Please type something in the input box.")

# --- Step-by-Step Breakdown ---
if user_input.strip() and 'logs' in locals():
    st.divider()
    st.subheader("3. Token-by-Token Processing Trace")
    st.write("Expand any step below to see how the mathematical gates reacted to that specific word:")

    for step in logs:
        with st.expander(f"Step {step['step']}: Word processed ➔ '{step['token']}'"):
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.metric(label="Reset Gate Active", value=str(step['reset']))
                st.caption("Lower values wipe out old context.")
                
            with c2:
                st.metric(label="Update Gate Active", value=str(step['update']))
                st.caption("Higher values mix in the new word.")
                
            with c3:
                st.metric(label="New Hidden State Memory", value=str(step['state']))
                st.caption("The saved memory vector after this step.")

# --- Quick Math Section ---
st.divider()
st.subheader("📐 The Core Logic Equations")
st.latex(r"h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t")
st.caption("This simple equation controls the balancing act: $(1 - z_t)$ is what we remember from the past, and $z_t$ is what we accept from the present.")