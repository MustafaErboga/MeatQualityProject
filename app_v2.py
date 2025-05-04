import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
import os # Model dosyasının varlığını kontrol etmek için (aslında LFS ile gerek kalmıyor ama emin olmak için kalabilir veya silebilirsiniz)

# Başlık
st.set_page_config(page_title="Meat Quality Classifier", layout="centered")
st.title("Et Kalitesi Sınıflandırma") # Başlığı önceki Türkçe haline çevirdim
st.write("Yüklediğiniz görselin kalitesini tahmin eder.") # Açıklamayı sadeleştirdim

# Modeli yükle
# Model dosyası artık Git LFS ile depoda yer alıyor, doğrudan dosyadan yüklenecek.
@st.cache_resource
def load_trained_model(model_path):
    st.info(f"Model yükleniyor: {model_path}") # Yükleme mesajını ekledim
    try:
        # Modeli doğrudan dosyadan yükle
        model = load_model(model_path)
        st.success("Model başarıyla yüklendi.") # Başarı mesajını ekledim
        return model
    except Exception as e:
        # Model yüklenemezse hata göster ve uygulamayı durdur
        st.error(f"Model yükleme hatası: {e}")
        st.stop()

# Model dosyasının adı ve yolu (GitHub deponuzdaki kök dizine göre)
# En son Git çıktınızda model adı 'meat_quality_cnn_model.h5' görünüyordu
MODEL_PATH_IN_REPO = "meat_quality_cnn_model.h5"

# Modeli yükleme fonksiyonunu çağır
model = load_trained_model(MODEL_PATH_IN_REPO)

# Sınıf İsimleri (Sizin kodunuzdaki ikili sınıflandırmaya göre)
class_names = ['Fresh', 'Spoiled']
# Eğer modeliniz 3 sınıfı (Fresh, Half-Fresh, Spoiled) tahmin ediyorsa ve
# çıkış katmanı 3 nöronluysa ve softmax kullanıyorsa, class_names'i güncelleyin
# ve tahmin mantığını (aşağıda) 3 sınıfa uygun hale getirin.
# Şu anki kodunuz ikili sınıflandırma çıktısı varsayıyor.

# Görsel yükleme
uploaded_file = st.file_uploader("Lütfen bir et fotoğrafı yükleyin (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"]) # Kapsamı genişlettim

if uploaded_file is not None:
    # Görseli göster
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Görseli modele uygun hale getir
    # Sizin kodunuzda 128x128 olarak yeniden boyutlandırılıyordu, onu korudum.
    img = image.resize((128, 128))

    # PIL görselini NumPy dizisine çevir ve normalize et (0-1 aralığına)
    img_array = img_to_array(img) / 255.0

    # Model batch input beklediği için boyut ekliyoruz (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin yap
    # Sizin kodunuzdaki ikili sınıflandırma tahmin mantığını korudum.
    # Model çıktısı tek bir değer (olasılık) varsayılıyor.
    prediction = model.predict(img_array) # prediction shape: (1, 1) veya (1,)
    predicted_probability = prediction[0][0] # Tahmin edilen tek değeri al

    # Sonucu yorumla (0.5 eşiğine göre)
    if predicted_probability > 0.5:
        predicted_class = 'Spoiled'
        confidence = predicted_probability * 100
    else:
        predicted_class = 'Fresh'
        confidence = (1 - predicted_probability) * 100 # Fresh olma olasılığı = 1 - Bozuk olma olasılığı

    # Sonuçları göster
    st.markdown("---")
    st.subheader("Tahmin Sonucu")
    st.write(f"**Sınıf:** {predicted_class}")
    st.write(f"**Güven:** {confidence:.2f}%")

    # Tahmin sonucuna göre mesaj göster
    if predicted_class == 'Fresh':
        st.success("Bu et görseli **taze** görünüyor.")
    else:
        st.error("Bu et görseli **bozulmuş** olabilir!")
