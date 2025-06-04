# app.py (Streamlit UI)
import streamlit as st
from detector.adversarial_detector import AdversarialDetector

def main():
    st.set_page_config(page_title="Adversarial Image Detector", layout="wide")
    st.title("Adversarial Image Detector")
    
    # Initialize detector
    detector = AdversarialDetector()
    
    # File upload
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
    with col2:
        if uploaded_file and st.button("Detect"):
            try:
                is_adv, distance = detector.detect(uploaded_file)
                st.subheader("Detection Results")
                
                if is_adv:
                    st.error("⚠️ **ADVERSARIAL DETECTED**")
                else:
                    st.success("✅ **NORMAL IMAGE**")
                    
                st.metric("Distance Score", f"{distance:.2f}")
                st.metric("Decision Threshold", f"{detector.threshold:.2f}")
                
                # Visual feedback
                progress_value = float(min(distance / detector.threshold, 1.0))
                st.progress(progress_value)
                
            except Exception as e:
                st.error(f"Detection failed: {str(e)}")

if __name__ == "__main__":
    main()