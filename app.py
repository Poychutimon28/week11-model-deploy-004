# app.py
# ==============================================================================
# โปรแกรมจำแนกโรค Covid-19 จากภาพ X-ray (Streamlit Web App)
# ------------------------------------------------------------------------------
# หมายเหตุสำคัญ (โปรดอ่านก่อนใช้งาน):
#   ไฟล์โมเดล .pkcls ทั้ง 3 ไฟล์ที่แนบมา (Logistic Regression, Neural Network,
#   Decision Tree) ถูกฝึกและบันทึกด้วยโปรแกรม "Orange Data Mining" โดยตัว
#   feature ที่ใช้ฝึก (n0, n1, n2, ...) มาจากการแปลงภาพ X-ray ผ่าน widget
#   "Image Embedding" ด้วยโมเดล embedder ชื่อ "SqueezeNet"
#
#   ดังนั้นตอนทำนายผล (predict) จำเป็นต้องแปลงภาพที่ผู้ใช้อัปโหลดด้วย
#   embedder ตัวเดียวกัน (SqueezeNet) ก่อนส่งเข้าโมเดลเสมอ ไม่เช่นนั้นตัวเลข
#   features จะไม่ตรงกับที่โมเดลเรียนรู้ไว้ ผลทำนายจะผิดทันที
#
#   ข้อกำหนด:
#   1) ต้องติดตั้งไลบรารี Orange3 และ Orange3-ImageAnalytics (ดู requirements.txt)
#   2) เครื่อง/เซิร์ฟเวอร์ที่รันแอปนี้ต้องต่ออินเทอร์เน็ตได้ เพราะ ImageEmbedder
#      ("SqueezeNet") ประมวลผลผ่าน embedding server ระยะไกลของ Orange
#      (api.garaza.io) ไม่ได้รันในเครื่องเอง ถ้าเซิร์ฟเวอร์นี้ล่มหรือถูกบล็อก
#      อินเทอร์เน็ตขาออก จะแปลงภาพเป็น embedding ไม่ได้
# ==============================================================================

import os

# บังคับให้ Qt (ที่ Orange3 ต้องใช้ผ่าน PyQt5 ตอน import) รันแบบ "offscreen"
# เพราะเซิร์ฟเวอร์ Streamlit Cloud ไม่มีจอแสดงผลจริง ต้องตั้งค่านี้ก่อน import
# Orange/PyQt5 ใด ๆ ไม่เช่นนั้นอาจเจอ error เกี่ยวกับการเปิดหน้าต่าง GUI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import glob
import tempfile
import joblib
import numpy as np
import streamlit as st

# พยายาม import Orange + Orange3-ImageAnalytics
ORANGE_AVAILABLE = False
ORANGE_IMPORT_ERROR = None
try:
    import Orange
    from Orange.data import Table
    from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
    ORANGE_AVAILABLE = True
except Exception as e:
    ORANGE_IMPORT_ERROR = f"{type(e).__name__}: {e}"

# ------------------------------------------------------------------------------
# 2) หัวข้อของแอป
# ------------------------------------------------------------------------------
st.set_page_config(page_title="จำแนกโรค Covid-19 จากภาพ X-ray", page_icon="🫁")
st.title("โปรแกรมจำแนกโรค covid จากภาพ x-ray")
st.caption(
    "อัปโหลดภาพเอกซเรย์ทรวงอก (Chest X-ray) แล้วกดปุ่ม 'ทำนายผล' "
    "ระบบจะจำแนกว่าภาพเข้าข่าย COVID-19, ปกติ (Normal) หรือ ปอดอักเสบ (Pneumonia)"
)

if not ORANGE_AVAILABLE:
    st.error(
        "ไม่สามารถ import ไลบรารีที่จำเป็นได้ กรุณาติดตั้งก่อนใช้งานด้วยคำสั่ง:\n\n"
        "`pip install -r requirements.txt`\n\n"
        "(ต้องมีทั้ง Orange3 และ Orange3-ImageAnalytics)"
    )
    st.code(ORANGE_IMPORT_ERROR or "ไม่ทราบสาเหตุ (ไม่มีข้อความ error)")
    st.stop()

# ------------------------------------------------------------------------------
# 1) ส่วนเลือกโมเดล + โหลดโมเดลด้วย joblib
# ------------------------------------------------------------------------------
MODEL_DIR = "models"
model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkcls")))

st.sidebar.header("เลือกโมเดล")

if not model_files:
    st.sidebar.error(
        f"ไม่พบไฟล์ .pkcls ในโฟลเดอร์ '{MODEL_DIR}/' "
        "กรุณานำไฟล์โมเดล (.pkcls) ไปวางไว้ในโฟลเดอร์นี้บน GitHub repo "
        "แล้ว deploy ใหม่อีกครั้ง"
    )
    st.stop()

model_path = st.sidebar.selectbox(
    "เลือกไฟล์โมเดล (.pkcls)",
    options=model_files,
    format_func=lambda p: os.path.basename(p),
)

st.sidebar.markdown("**หรืออัปโหลดไฟล์โมเดล (.pkcls)**")
uploaded_model = st.sidebar.file_uploader(
    "เลือกไฟล์โมเดล (.pkcls)", type=["pkcls"], label_visibility="collapsed"
)
if uploaded_model is not None:
    tmp_model_path = os.path.join(tempfile.gettempdir(), uploaded_model.name)
    with open(tmp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    model_path = tmp_model_path


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path: str):
    """โหลดโมเดล Orange (.pkcls) ด้วย joblib และ cache ไว้ไม่ให้โหลดซ้ำทุกครั้ง"""
    return joblib.load(path)


try:
    model = load_model(model_path)
except Exception as e:
    st.sidebar.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

st.sidebar.success(f"โหลดโมเดล '{os.path.basename(model_path)}' สำเร็จ")

# ------------------------------------------------------------------------------
# 3) ตัว embedder (SqueezeNet) — cache ไว้ ไม่ต้องสร้างใหม่ทุกครั้ง
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    # "squeezenet" ต้องตรงกับที่เลือกไว้ตอนฝึกใน widget "Image Embedding" ของ Orange
    return ImageEmbedder(model="squeezenet")


# ------------------------------------------------------------------------------
# 4) ส่วนอัปโหลดภาพ X-ray
# ------------------------------------------------------------------------------
st.subheader("อัปโหลดภาพ X-ray")

uploaded_image = st.file_uploader(
    "เลือกไฟล์ภาพ (jpg, jpeg, png)", type=["jpg", "jpeg", "png"]
)

if uploaded_image is None:
    st.info("กรุณาอัปโหลดภาพ X-ray เพื่อเริ่มการทำนาย")
    st.stop()

st.image(uploaded_image, caption="ภาพที่อัปโหลด", use_container_width=True)

if st.button("ทำนายผล", type="primary"):
    tmp_image_path = None
    try:
        # ------------------------------------------------------------------
        # 4.1) บันทึกภาพลงไฟล์ชั่วคราว เพราะ ImageEmbedder ต้องการ path ของไฟล์
        # ------------------------------------------------------------------
        suffix = os.path.splitext(uploaded_image.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_image.getbuffer())
            tmp_image_path = tmp_file.name

        # ------------------------------------------------------------------
        # 4.2) แปลงภาพเป็นเวกเตอร์ตัวเลข (embedding) ด้วย SqueezeNet
        #      ขั้นตอนนี้ต้องต่ออินเทอร์เน็ตได้ (เรียก embedding server ของ Orange)
        # ------------------------------------------------------------------
        with st.spinner("กำลังแปลงภาพเป็น embedding (ต้องใช้อินเทอร์เน็ต)..."):
            embedder = get_embedder()
            embeddings = embedder([tmp_image_path])

        embedding_vector = embeddings[0] if embeddings is not None else None
        if embedding_vector is None:
            st.error(
                "แปลงภาพเป็น embedding ไม่สำเร็จ "
                "(อาจเชื่อมต่อ embedding server ไม่ได้ หรือไฟล์ภาพเสียหาย) "
                "กรุณาลองใหม่อีกครั้ง"
            )
            st.stop()

        # ------------------------------------------------------------------
        # 4.3) สร้าง Orange Table จาก domain เดิมของโมเดล แล้วทำนายผล
        # ------------------------------------------------------------------
        domain = model.domain
        X = np.array([embedding_vector], dtype=float)

        if X.shape[1] != len(domain.attributes):
            st.error(
                f"จำนวนมิติของ embedding ({X.shape[1]}) ไม่ตรงกับจำนวน feature "
                f"ที่โมเดลนี้ฝึกไว้ ({len(domain.attributes)}) "
                "แสดงว่า embedder ที่ใช้ตอนฝึกอาจไม่ใช่ SqueezeNet "
                "กรุณาตรวจสอบไฟล์ workflow (.ows) อีกครั้ง"
            )
            st.stop()

        instance_table = Table.from_numpy(domain, X)

        pred_idx, probs = model(instance_table, ret=Orange.classification.Model.ValueProbs)

        class_var = domain.class_var
        predicted_label = class_var.values[int(pred_idx[0])]
        confidence = float(np.max(probs[0])) * 100

        # ------------------------------------------------------------------
        # 4.4) แสดงผลการทำนาย
        # ------------------------------------------------------------------
        st.success(f"ผลการทำนาย: **{predicted_label}** (ความมั่นใจ {confidence:.2f}%)")

        st.write("ความน่าจะเป็นของแต่ละคลาส:")
        prob_dict = {
            class_var.values[i]: f"{p * 100:.2f}%"
            for i, p in enumerate(probs[0])
        }
        st.table(prob_dict)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างทำนายผล: {e}")

    finally:
        # ลบไฟล์ชั่วคราวทิ้งเสมอ ไม่ว่าจะสำเร็จหรือ error
        if tmp_image_path and os.path.exists(tmp_image_path):
            os.remove(tmp_image_path)
