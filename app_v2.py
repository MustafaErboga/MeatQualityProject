import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

# Başlık
st.set_page_config(page_title="Meat Quality Classifier", layout="centered")
st.title(" Meat Quality Classifier")
st.write("Yüklediğiniz görselin **Fresh (taze)** mi yoksa **Spoiled (bozuk)** mu olduğunu tahmin eder.")

# Modeli yükle
@st.cache_resource
def load_trained_model():
    return load_model("meat_quality_cnn_model.h5")

model = load_trained_model()
class_names = ['Fresh', 'Spoiled']

# Görsel yükleme
uploaded_file = st.file_uploader("Lütfen bir et fotoğrafı yükleyin (.jpg)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli göster
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Görseli modele uygun hale getir
    img = image.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin
    prediction = model.predict(img_array)
    predicted_probability = prediction[0][0]

    if predicted_probability > 0.5:
        predicted_class = 'Spoiled'
        confidence = predicted_probability * 100
    else:
        predicted_class = 'Fresh'
        confidence = (1 - predicted_probability) * 100

    # Sonuç
    st.markdown("---")
    st.subheader(" Tahmin Sonucu")
    st.write(f"**Sınıf:** {predicted_class}")
    st.write(f"**Güven:** {confidence:.2f}%")

    if predicted_class == 'Fresh':
        st.success("Bu et görseli **taze** görünüyor.")
    else:
        st.error("Bu et görseli **bozulmuş** olabilir!")
=======
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

# Başlık
st.set_page_config(page_title="Meat Quality Classifier", layout="centered")
st.title(" Meat Quality Classifier")
st.write("Yüklediğiniz görselin **Fresh (taze)** mi yoksa **Spoiled (bozuk)** mu olduğunu tahmin eder.")

# Modeli yükle
@st.cache_resource
def load_trained_model():
    return load_model("meat_quality_cnn_model.h5")

model = load_trained_model()
class_names = ['Fresh', 'Spoiled']

# Görsel yükleme
uploaded_file = st.file_uploader("Lütfen bir et fotoğrafı yükleyin (.jpg)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli göster
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Görseli modele uygun hale getir
    img = image.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin
    prediction = model.predict(img_array)
    predicted_probability = prediction[0][0]

    if predicted_probability > 0.5:
        predicted_class = 'Spoiled'
        confidence = predicted_probability * 100
    else:
        predicted_class = 'Fresh'
        confidence = (1 - predicted_probability) * 100

    # Sonuç
    st.markdown("---")
    st.subheader(" Tahmin Sonucu")
    st.write(f"**Sınıf:** {predicted_class}")
    st.write(f"**Güven:** {confidence:.2f}%")

    if predicted_class == 'Fresh':
        st.success("Bu et görseli **taze** görünüyor.")
    else:
        st.error("Bu et görseli **bozulmuş** olabilir!")
