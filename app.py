import streamlit as st
from SA_utils import load_m_n_v, manual_testing,wp,output_lable
import time

# Load the model and vectorizer
model, vectorizer = load_m_n_v()

# Streamed response emulator
def response_generator(user_input):
    # Clean the input text
    cleaned_text = manual_testing(user_input)
    print(f"Cleaned text: {cleaned_text}")  # Debugging print statement

    # Ensure cleaned_text is a string
    if not isinstance(cleaned_text, str):
        cleaned_text = str(cleaned_text)

    # Transform the cleaned text using the vectorizer
    k_text_vectorized = vectorizer.transform([cleaned_text])
    print(f"Vectorized text shape: {k_text_vectorized.shape}")  # Debugging print statement

    # Predict sentiment
    predict = model.predict(k_text_vectorized)

    if predict[0] == 0:
        sentiment = "Negative"
    else:
        sentiment = "Positive"

    response = f"The sentiment of your message is: {sentiment}"
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.title("Chatbot with Sentiment Analysis")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response_parts = response_generator(prompt)
        response = ''.join(response_parts)
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
