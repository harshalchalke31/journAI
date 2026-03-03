import streamlit as st
from apps import app1, app2  # Import the individual app modules

# Page Configurations
st.set_page_config(
    page_title="JournAI Suite",
    page_icon="📝",
    layout="centered",
)

# Sidebar Branding
st.sidebar.image("assets/logo1.webp", use_container_width=True, caption="JournAI")  # Optional logo
st.sidebar.markdown("## Navigation")
option = st.sidebar.radio("Go to:", ["Home", "App 1", "App 2"])

# Homepage Content
if option == "Home":
    st.title("Welcome to JournAI")
    st.write("A suite of AI-powered tools at your fingertips.")
    st.markdown(
        """
        - **App 1**: A powerful tool for [specific task].
        - **App 2**: Another tool for [specific task].
        """
    )
    st.info("Use the sidebar to navigate between apps.")

# Navigate to App 1
elif option == "App 1":
    app1.run()

# Navigate to App 2
elif option == "App 2":
    app2.run()
