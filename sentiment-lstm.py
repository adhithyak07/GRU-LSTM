import streamlit as st
import numpy as np

# --- Page Setup ---
st.set_page_config(
    page_title="LSTM Sentiment Analyzer",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 LSTM Sentiment Analysis Simulator")
st.write("See how an LSTM processes a sequence of words, manages its long-term memory, and calculates a final positive or negative sentiment score.")
st.divider()

# --- Sentiment Word Dictionary (Simulated Word Embeddings) ---
VALENCE_DICT = {
    # Positive words
    "love": 0.8, "great": 0.7, "fantastic": 0.9, "incredible": 0.9, "good": 0.5, 
    "brilliant": 0.8, "happy": 0.6, "best": 0.7, "enjoyed": 0.6, "recommend": 0.5,
    # Negative words
    "hate": -0.8, "bad": -0.6, "terrible": -0.9, "worst": -0.9, "boring": -0.5, 
    "poor": -0.6, "waste": -0.8, "awful": -0.7, "disappointed": -0.7, "stupid": -0.5
}

# --- LSTM Engine ---
def analyze_sentiment_lstm(text):
    tokens = [t.strip().lower().strip(".,!?\"'") for t in text.split() if t.strip()]
    if not tokens:
        return 0.5, []
    
    # 1-Dimensional tracking for maximum clarity:
    # c_t = Long-term sentiment accumulation
    # h_t = Short-term immediate sentiment focus
    c_t = 0.0  
    h_t = 0.0  
    history = []
    
    for idx, token in enumerate(tokens):
        # Look up word valence (default to 0 if neutral/unknown)
        word_valence = VALENCE_DICT.get(token, 0.0)
        
        # Simulate LSTM Gate Mathematics based on text context
        # 1. Forget Gate: If a word is highly opinionated, lower forget gate to clear previous mixed signals
        forget_gate = 0.3 if abs(word_valence) > 0.5 else 0.85
        
        # 2. Input Gate: How much value this specific word injects into memory
        input_gate = 0.7 if word_valence != 0.0 else 0.1
        
        # Update long-term memory (Cell State)
        old_c_t = c_t
        c_t = (forget_gate * old_c_t) + (input_gate * word_valence)
        c_t = np.clip(c_t, -1.0, 1.0)
        
        # 3. Output Gate: Determines the visible short-term hidden state
        output_gate = 0.8
        h_t = output_gate * np.tanh(c_t)
        
        history.append({
            "step": idx + 1,
            "token": token,
            "valence": word_valence,
            "forget": round(forget_gate, 2),
            "input": round(input_gate, 2),
            "cell_state": round(c_t, 3),
            "hidden_state": round(h_t, 3)
        })
        
    # Convert final hidden state into a 0% to 100% sentiment score via Sigmoid
    final_score = 1 / (1 + np.exp(-h_t * 3))
    return final_score, history

# --- Layout ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Input Review Text")
    user_input = st.text_area(
        "Enter a product or movie review:",
        value="The acting was fantastic but the script was a terrible waste.",
        height=100
    )
    
    st.markdown("""
    ### 💡 Try these sentence combinations:
    1. **Shift in sentiment:** *"The movie started out bad, but the ending was great."* 2. **Neutral words buffering:** *"The actual package was blue and boxy, but I love it."*
    """)

with col_right:
    st.subheader("2. Sentiment Verdict")
    
    if user_input.strip():
        score, logs = analyze_sentiment_lstm(user_input)
        
        # Display large status color card based on prediction
        if score >= 0.55:
            st.success(f"🟢 **POSITIVE SENTIMENT**\n\nOverall Rating Score: {score:.1%}")
        elif score <= 0.45:
            st.error(f"🔴 **NEGATIVE SENTIMENT**\n\nOverall Rating Score: {score:.1%}")
        else:
            st.warning(f"🟡 **NEUTRAL / MIXED SENTIMENT**\n\nOverall Rating Score: {score:.1%}")
    else:
        st.info("Awaiting text token streams...")

# --- Step-by-Step State Breakdown ---
if user_input.strip() and 'logs' in locals():
    st.divider()
    st.subheader("3. Token Pipeline Inspection Track")
    st.write("See how each word alters the long-term memory cell state ($C_t$):")

    for step in logs:
        # Give specific styling to known sentiment drivers
        label_suffix = ""
        if step['valence'] > 0: label_suffix = " 🟢 (+)"
        elif step['valence'] < 0: label_suffix = " 🔴 (-)"
            
        with st.expander(f"Step {step['step']}: '{step['token']}'{label_suffix}"):
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.metric(label="Word Value", value=str(step['valence']))
                st.caption("Extracted text vector valence.")
                
            with c2:
                st.metric(label="Forget Gate", value=str(step['forget']))
                st.caption("How much past text history is kept.")
                
            with c3:
                st.metric(label="Input Gate", value=str(step['input']))
                st.caption("How much weight this word carries.")
                
            with c4:
                st.metric(label="Long-Term Memory", value=str(step['cell_state']))
                st.caption("Updated cell state ($C_t$).")

# --- Math Overview ---
st.divider()
st.subheader("📐 The Sentiment Balance Equation")
st.latex(r"C_t = (\text{Forget Gate} \times C_{t-1}) + (\text{Input Gate} \times \text{New Sentiment})")
st.caption("If the LSTM encounters a word like 'but' or a sudden strong opinion, the Forget Gate shrinks to clear out conflicting history, allowing the Input Gate to overwrite the network with the new sentiment trajectory.")