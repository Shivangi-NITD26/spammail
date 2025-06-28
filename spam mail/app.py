import streamlit as st
import pickle

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error("Model loading failed.")
        st.error("Make sure 'model.pkl' and 'vectorizer.pkl' are in the app folder.")
        return None, None

def main():
    st.title("📧 Email Spam Detector")
    st.write("Enter an email message below to check whether it's spam or not.")

    model, vectorizer = load_model()

    if model and vectorizer:
        user_input = st.text_area("Your email text", "Win a free iPhone! Click here to claim now.")

        if st.button("Predict"):
            if not user_input.strip():
                st.warning("Please type something first!")
            else:
                try:
                    input_vector = vectorizer.transform([user_input])
                    prediction = model.predict_proba(input_vector)[0]
                    spam_score = prediction[0]
                    ham_score = prediction[1]

                    if spam_score > 0.5:
                        st.error(f"🚨 Spam detected! Confidence: {spam_score * 100:.1f}%")
                    else:
                        st.success(f"✅ Not spam. Confidence: {ham_score * 100:.1f}%")

                    st.progress(spam_score)

                    with st.expander("Why this result?"):
                        st.write(f"Spam: {spam_score:.4f}")
                        st.write(f"Ham: {ham_score:.4f}")

                        if hasattr(model, 'coef_'):
                            words = vectorizer.get_feature_names_out()
                            weights = model.coef_[0]
                            word_ids = input_vector.nonzero()[1]

                            if word_ids.size > 0:
                                st.write("Most influential words:")
                                important = sorted(
                                    [(words[i], weights[i]) for i in word_ids],
                                    key=lambda x: abs(x[1]),
                                    reverse=True
                                )[:5]

                                for word, weight in important:
                                    label = "spam" if weight > 0 else "ham"
                                    st.write(f"- **{word}** → {label} (score: {weight:.2f})")
                except Exception as e:
                    st.error(f"Oops! Something went wrong: {e}")

if __name__ == "__main__":
    main()
