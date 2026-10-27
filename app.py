import os
import gradio as gr
import json
import numpy as np
from tensorflow.keras.models import load_model

# Nạp mô hình AI
model = load_model("animals_classifier.h5")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

# Từ điển thông tin sinh học (giữ nguyên phần từ điển của bạn ở đây...)
BIO_DATABASE = {
    "cane": {"en": "dog", "scientific": "Canis lupus familiaris", "vn": "Chó", "iucn": "Ít quan tâm", "habitat": "Gần con người", "behavior": "Trung thành", "lifespan": "10-13 năm"},
    "cavallo": {"en": "horse", "scientific": "Equus caballus", "vn": "Ngựa", "iucn": "Thuần hóa", "habitat": "Đồng cỏ", "behavior": "Sống bầy đàn", "lifespan": "25-30 năm"},
    "elefante": {"en": "elephant", "scientific": "Loxodonta africana", "vn": "Voi", "iucn": "Nguy cấp", "habitat": "Rừng/Thảo nguyên", "behavior": "Thông minh, sống mẫu hệ", "lifespan": "60-70 năm"},
    "farfalla": {"en": "butterfly", "scientific": "Rhopalocera", "vn": "Bướm", "iucn": "Ít quan tâm", "habitat": "Vườn hoa", "behavior": "Thụ phấn", "lifespan": "Vài tuần"},
    "gallina": {"en": "chicken", "scientific": "Gallus gallus", "vn": "Gà", "iucn": "Thuần hóa", "habitat": "Trang trại", "behavior": "Ăn tạp", "lifespan": "5-10 năm"},
    "gatto": {"en": "cat", "scientific": "Felis catus", "vn": "Mèo", "iucn": "Ít quan tâm", "habitat": "Gia đình", "behavior": "Độc lập, nhanh nhẹn", "lifespan": "12-15 năm"},
    "mucca": {"en": "cow", "scientific": "Bos taurus", "vn": "Bò", "iucn": "Thuần hóa", "habitat": "Đồng cỏ", "behavior": "Hiền lành, ăn cỏ", "lifespan": "15-20 năm"},
    "pecora": {"en": "sheep", "scientific": "Ovis aries", "vn": "Cừu", "iucn": "Thuần hóa", "habitat": "Đồi cỏ", "behavior": "Tính bầy đàn cao", "lifespan": "10-12 năm"},
    "ragno": {"en": "spider", "scientific": "Araneae", "vn": "Nhện", "iucn": "Ít quan tâm", "habitat": "Khắp nơi", "behavior": "Giăng tơ bắt mồi", "lifespan": "1-2 năm"},
    "scoiattolo": {"en": "squirrel", "scientific": "Sciurus vulgaris", "vn": "Sóc", "iucn": "Ít quan tâm", "habitat": "Rừng cây", "behavior": "Leo trèo nhanh", "lifespan": "3-7 năm"},
}

def predict_animal(img):
    if img is None:
        return "⚠️ Vui lòng tải lên một bức ảnh động vật!"
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]
    idx = np.argmax(pred)
    species_key = idx_to_class.get(idx, "unknown")
    confidence = float(pred[idx] * 100)

    info = BIO_DATABASE.get(species_key, {
        "en": species_key, "vn": "Không rõ", "scientific": "N/A", 
        "iucn": "N/A", "habitat": "N/A", "behavior": "N/A", "lifespan": "N/A"
    })

    result_text = f"""
    🏷️ Tên loài (EN - VN) : {info['en'].upper()} ({info['vn']})
    🧬 Tên khoa học       : {info['scientific']}
    📊 Độ tin cậy         : {confidence:.2f}%
    --------------------------------------------------------
    🛡️ Tình trạng IUCN    : {info['iucn']}
    🏡 Môi trường sống    : {info['habitat']}
    🐾 Tập tính           : {info['behavior']}
    ⏳ Tuổi thọ           : {info['lifespan']}
    """
    return result_text

demo = gr.Interface(
    fn=predict_animal,
    inputs=gr.Image(type="pil", label="Tải ảnh động vật lên đây"),
    outputs=gr.Textbox(label="Kết quả nhận diện & Thông tin sinh học"),
    title="🐾 Ứng Dụng AI Nhận Diện Động Vật & Tra Cứu Sinh học",
    description="Hệ thống AI nhận diện loài và cung cấp thông tin sinh học chi tiết."
)

if __name__ == "__main__":
    # Lấy cổng PORT do Render tự động cấp phát
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)