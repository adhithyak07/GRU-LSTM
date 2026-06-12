import streamlit as st
import numpy as np
import time

# --- Page Setup ---
st.set_page_config(
    page_title="Bi-LSTM Simple Explainer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Bidirectional LSTM Explainer")
st.write("See how a Bidirectional LSTM (Bi-LSTM) reads text both forward and backward to catch full context.")
st.divider()

# --- Bi-LSTM Simulation Engine ---
def run_simple_bilstm(text):
    tokens = [t.strip() for t in text.split() if t.strip()]
    if not tokens:
        return 0.5, []
    
    # 1. FORWARD PASS: Left to Right
    h_forward = np.zeros(2)
    forward_states = []
    
    for idx, token in enumerate(tokens):
        seed = (sum(ord(c) for c in token) + 5) % 100
        np.random.seed(seed)
        
        # Simple processing layer
        clean_token = token.lower().strip(".,!?\"'")
        bias = 0.5 if clean_token in ['fantastic', 'incredible', 'great', 'love', 'good'] else 0.0
        if clean_token in ['bad', 'terrible', 'worst', 'boring', 'poor']: bias = -0.5
        
        h_forward = 0.6 * h_forward + 0.4 * np.tanh(np.random.normal(bias, 0.1, 2))
        forward_states.append(np.round(h_forward, 2))
        
    # 2. BACKWARD PASS: Right to Left
    h_backward = np.zeros(2)
    backward_states = [None] * len(tokens)
    
    for idx in reversed(range(len(tokens))):
        token = tokens[idx]
        seed = (sum(ord(c) for c in token) + 12) % 100  # Different seed for backward layer
        np.random.seed(seed)
        
        clean_token = token.lower().strip(".,!?\"'")
        bias = 0.5 if clean_token in ['fantastic', 'incredible', 'great', 'love', 'good'] else 0.0
        if clean_token in ['bad', 'terrible', 'worst', 'boring', 'poor']: bias = -0.5
        
        h_backward = 0.6 * h_backward + 0.4 * np.tanh(np.random.normal(bias, 0.1, 2))
        backward_states[idx] = np.round(h_backward, 2)

    # 3. COMBINE BOTH PASSES
    history = []
    for idx, token in enumerate(tokens):
        # Concatenate forward and backward vectors to make a 4-element combined state
        combined_state = np.concatenate([forward_states[idx], backward_states[idx]])
        history.append({
            "step": idx + 1,
            "token": token,
            "forward": forward_states[idx],
            "backward": backward_states[idx],
            "combined": combined_state
        })
        
    # Calculate final score based on the combined final boundaries
    final_vector = np.concatenate([forward_states[-1], backward_states[0]])
    final_score = 1 / (1 + np.exp(-np.mean(final_vector) * 3))
    return final_score, history

# --- Layout Split ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Enter Your Text")
    user_input = st.text_input(
        "Type a sentence to analyze:",
        value="Although it started out terrible, the climax became fantastic."
    )
    
    st.markdown("""
    ### Why use Bidirectional layers?
    In sentences like *"Although it started out **terrible**, the climax became **fantastic**,"* standard models might get confused early on. 
    * The **Forward layer** tracks how the sentence *progresses*.
    * The **Backward layer** loops backwards to track how the sentence *ends*, giving clues about the ultimate context beforehand.
    """)

with col_right:
    st.subheader("2. Combined Prediction")
    
    if user_input.strip():
        score, logs = run_simple_bilstm(user_input)
        
        if score >= 0.5:
            st.success(f"🟢 **Positive Sentiment** (Confidence: {score:.1%})")
        else:
            st.error(f"🔴 **Negative Sentiment** (Confidence: {(1 - score):.1%})")
    else:
        st.info("Please type something in the input box.")

# --- Step-by-Step Breakdown ---
if user_input.strip() and 'logs' in locals():
    st.divider()
    st.subheader("3. Bidirectional Token-by-Token Processing Trace")
    st.write("Notice how every word has insights from both directions at the exact same time:")

    for step in logs:
        with st.expander(f"Step {step['step']}: Word processed ➔ '{step['token']}'"):
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.metric(label="➡️ Forward Pass State", value=str(step['forward']))
                st.caption("Context collected from left-to-right up to this word.")
                
            with c2:
                st.metric(label="⬅️ Backward Pass State", value=str(step['backward']))
                st.caption("Context collected from right-to-left up to this word.")
                
            with c3:
                st.metric(label="🔄 Combined Vector (Concatenated)", value=str(step['combined']))
                st.caption("The full bidirectional signature for this token slot.")

# --- Quick Math Section ---
st.divider()
st.subheader("📐 The Core Combination Logic")
st.latex(r"h_t = [\,\vec{h}_t \,,\, \overleftarrow{h}_t\,]")
st.caption("Instead of running complex algorithms to merge tracks, a Bi-LSTM simply attaches (concatenates) the Forward Vector and Backward Vector together side-by-side.")