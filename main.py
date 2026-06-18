import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict(input_data):
    input_data = [f"{input_data['title']} {input_data['subject']} {input_data['text']}"]
    input_data = vectorizer.transform(input_data)
    return model.predict(input_data)

def main():
    st.set_page_config(page_title="Fake News Detection", page_icon="📰")
    st.title("Fake News Detection Dashboard")
    st.write("Masukkan judul, subjek, dan teks berita untuk memeriksa apakah berita tersebut kemungkinan nyata atau palsu.")

    title = st.text_input("Judul Berita")
    subject = st.text_input("Subjek Berita")
    text = st.text_area("Teks Berita", height=250)

    if st.button("Deteksi"):
        if not title.strip() or not subject.strip() or not text.strip():
            st.warning("Silakan masukkan judul, subjek, dan teks berita terlebih dahulu.")
        else:
            try:
                payload = {"title": title, "subject": subject, "text": text}
                label = predict(payload)[0]
                st.write(f"Prediksi: **{label}**")
                
                if label == 1:
                    st.warning("Berita tersebut kemungkinan palsu.")
                    
                else:
                    st.success("Berita tersebut kemungkinan nyata.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memprediksi: {e}")


if __name__ == "__main__":
    main()
    
