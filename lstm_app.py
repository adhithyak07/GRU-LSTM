import streamlit as st
import numpy as np
import time

# --- Page Setup ---
st.set_page_config(
    page_title="LSTM Simple Explainer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 LSTM Step-by-Step Explainer")
st.write("See how a Long Short-Term Memory (LSTM) network uses three independent gates to manage text sequences.")
st.divider()

# --- LSTM Simulation Engine ---
def run_simple_lstm(text):
    tokens = [t.strip() for t in text.split() if t.strip()]
    if not tokens:
        return None, 0.5, []
    
    # LSTMs track BOTH a Hidden State (h_t) and a Cell State (C_t)
    h_t = np.zeros(2)  # Short-term memory
    c_t = np.zeros(2)  # Long-term memory
    history = []
    
    for idx, token in enumerate(tokens):
        # Create a predictable pseudo-random state based on the word
        seed = sum(ord(c) for c in token) % 100
        np.random.seed(seed)
        
        # LSTM uses 3 distinct gates
        forget_gate = np.random.uniform(0.1, 0.9, 2)
        input_gate = np.random.uniform(0.1, 0.9, 2)
        output_gate = np.random.uniform(0.1, 0.9, 2)
        
        # Sentiment score lookup
        clean_token = token.lower().strip(".,!?\"'")
        bias = 0.0
        if clean_token in ['fantastic', 'incredible', 'great', 'love', 'good']:
            bias = 0.6
        elif clean_token in ['bad', 'terrible', 'worst', 'boring', 'poor']:
            bias = -0.6
            
        # LSTM Processing Pipeline
        candidate_cell = np.tanh(np.random.normal(bias, 0.1, 2))
        
        # 1. Update Long-term memory (Cell State)
        c_t = (forget_gate * c_t) + (input_gate * candidate_cell)
        c_t = np.clip(c_t, -1.0, 1.0)
        
        # 2. Update Short-term memory (Hidden State)
        h_t = output_gate * np.tanh(c_t)
        
        history.append({
            "step": idx + 1,
            "token": token,
            "forget": np.round(forget_gate, 2),
            "input": np.round(input_gate, 2),
            "output": np.round(output_gate, 2),
            "cell_state": np.round(c_t, 2),
            "hidden_state": np.round(h_t, 2)
        })
        
    final_score = 1 / (1 + np.exp(-np.mean(h_t) * 3))
    return final_score, history

# --- Layout Split ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Enter Your Text")
    user_input = st.text_input(
        "Type a sentence to analyze:",
        value="The graphics were fantastic, but the gameplay loop was boring."
    )
    
    st.markdown("""
    ### How does an LSTM manage memory?
    * **Forget Gate ($f_t$):** Controls how much of the old *long-term* memory to throw away.
    * **Input Gate ($i_t$):** Controls what new information gets added to the long-term memory.
    * **Output Gate ($o_t$):** Filters the long-term memory to create the new *short-term* hidden state.
    """)

with col_right:
    st.subheader("2. Final Inference Result")
    
    if user_input.strip():
        score, logs = run_simple_lstm(user_input)
        
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
    st.write("Expand any step below to see how the three mathematical gates altered internal states:")

    for step in logs:
        with st.expander(f"Step {step['step']}: Word processed ➔ '{step['token']}'"):
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.metric(label="Forget / Input Gates", value=f"{step['forget']} / {step['input']}")
                st.caption("Balance between erasing old vs writing new history.")
                
            with c2:
                st.metric(label="Output Gate Active", value=str(step['output']))
                st.caption("Determines what context passes to the output hidden state.")
                
            with c3:
                st.metric(label="Cell State (Long Term)", value=str(step['cell_state']))
                st.caption("The internal long-term vector track ($C_t$).")

# --- Quick Math Section ---
st.divider()
st.subheader("📐 Core LSTM Logic Equations")
st.latex(r"C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t")
st.latex(r"h_t = o_t \odot \tanh(C_t)")
st.caption("Notice how the Cell State ($C_t$) acts like a conveyor belt, allowing information to pass down the sequence completely unchanged if the forget gate ($f_t$) is wide open.")