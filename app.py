# app.py
# ============================================================================
# โปรแกรมจำแนกโรค Covid-19 จากภาพ X-ray (Streamlit Web App)
# ----------------------------------------------------------------------------
# หมายเหตุสำคัญ (โปรดอ่านก่อนใช้งาน):
#   ไฟล์โมเดล .pkcls ทั้ง 3 ไฟล์ (Logistic Regression, Neural Network,
#   Decision Tree) ถูกฝึกและบันทึกด้วยโปรแกรม "Orange Data Mining" โดยแปลง
#   ภาพ X-ray เป็นตัวเลข (features ชื่อ n0, n1, n2, ...) ผ่าน widget
#   "Image Analytics" แบบ SqueezeNet embedding ไม่ใช่ตัวเลข/หมวดหมู่ที่มนุษย์
#   กรอกเองได้ตรง ๆ ดังนั้นแอปนี้จึงให้ผู้ใช้ "อัปโหลดภาพ X-ray" แล้วระบบจะ
#   แปลงภาพเป็นตัวเลขชุดเดียวกันนี้ให้อัตโนมัติก่อนส่งเข้าโมเดล
#
#   ข้อกำหนด: ต้องติดตั้งไลบรารี Orange3 และ Orange3-ImageAnalytics ไว้ใน
#   เครื่อง/เซิร์ฟเวอร์ที่รันแอปนี้ด้วย (ดูไฟล์ requirements.txt) ไม่เช่นนั้น
#   joblib.load หรือการแปลงภาพจะไม่สำเร็จ
# ============================================================================

import os
import glob
import joblib
import numpy as np
import streamlit as st

# พยายาม import Orange (จำเป็นสำหรับ unpickle โมเดล .pkcls)
# หมายเหตุ: ดักจับ Exception แบบกว้าง (ไม่ใช่แค่ ImportError) แล้วเก็บข้อความ
# error จริงไว้แสดงผล เพราะบางครั้ง Orange3 อาจ import ไม่สำเร็จด้วยสาเหตุอื่น
# ที่ไม่ใช่ "ไม่มีไลบรารี" ตรง ๆ (เช่น ขาด dependency ย่อยบางตัว) การเห็น
# ข้อความ error จริงจะช่วยวินิจฉัยปัญหาได้แม่นยำกว่า
ORANGE_AVAILABLE = False
ORANGE_IMPORT_ERROR = None
try:
    import Orange
    from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
    ORANGE_AVAILABLE = True
except Exception as e:
    ORANGE_IMPORT_ERROR = f"{type(e).__name__}: {e}"

# พยายาม import ตัวแปลงภาพเป็น embedding vector (Orange3-ImageAnalytics)
# ใช้แปลงภาพ X-ray ที่ผู้ใช้อัปโหลดให้เป็นตัวเลข (embedding) ชุดเดียวกับ
# ตอนฝึกโมเดล (SqueezeNet) ก่อนส่งเข้าโมเดลเพื่อทำนายผล
IMAGE_EMBEDDER_AVAILABLE = False
IMAGE_EMBEDDER_IMPORT_ERROR = None
try:
    from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
    IMAGE_EMBEDDER_AVAILABLE = True
except Exception as e:
    IMAGE_EMBEDDER_IMPORT_ERROR = f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------------
# 2) หัวข้อของแอป
# ----------------------------------------------------------------------------
st.set_page_config(page_title="จำแนกโรค Covid-19 จากภาพ X-ray", page_icon="🩻")
st.title("โปรแกรมจำแนกโรค Covid-19 จากภาพ X-ray")
st.caption(
    "อัปโหลดภาพเอกซเรย์ทรวงอก (Chest X-ray) แล้วกดปุ่ม 'ทำนายผล' "
    "ระบบจะจำแนกว่าภาพเข้าข่าย COVID-19, ปกติ (Normal) หรือ ปอดอักเสบ (Pneumonia)"
)

if not ORANGE_AVAILABLE:
    st.error(
        "ไม่สามารถ import ไลบรารี Orange3 ได้ กรุณาติดตั้งก่อนใช้งานด้วยคำสั่ง:\n\n"
        "`pip install -r requirements.txt`\n\n"
        "(โมเดล .pkcls ในโปรเจกต์นี้ถูกฝึกด้วยโปรแกรม Orange Data Mining "
        "จึงต้องใช้ไลบรารี Orange3 ในการโหลดโมเดล)"
    )
    # แสดงข้อความ error จริงเพื่อช่วยวินิจฉัยปัญหา (เช่น dependency ที่ขาดหาย)
    st.code(ORANGE_IMPORT_ERROR or "ไม่ทราบสาเหตุ (ไม่มีข้อความ error)")
    st.stop()


# ----------------------------------------------------------------------------
# 1) ส่วนเลือกโมเดล + โหลดโมเดลด้วย joblib
#    ผู้ใช้เลือกได้ทั้งจากไฟล์ในโฟลเดอร์ models/ หรืออัปโหลดไฟล์ .pkcls เอง
# ----------------------------------------------------------------------------
MODEL_DIR = "models"  # โฟลเดอร์ที่เก็บไฟล์โมเดล .pkcls บน repo (ระดับเดียวกับ app.py)

# ค้นหาไฟล์ .pkcls ทั้งหมดในโฟลเดอร์ที่กำหนด
model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkcls")))

st.sidebar.header("เลือกโมเดล")

model_choice_name = st.sidebar.selectbox(
    "เลือกไฟล์โมเดล (.pkcls)",
    options=[os.path.basename(p) for p in model_files] if model_files else [],
    placeholder="ไม่พบโมเดลในโฟลเดอร์ models/",
) if model_files else None

st.sidebar.markdown("**หรืออัปโหลดไฟล์โมเดล (.pkcls)**")
uploaded_model = st.sidebar.file_uploader(
    "อัปโหลดไฟล์โมเดลของคุณเอง",
    type=["pkcls"],
    label_visibility="collapsed",
)

# กำหนด path ของโมเดลที่จะใช้จริง: อัปโหลดเองมีความสำคัญกว่าถ้ามีการอัปโหลด
model_path = None
if uploaded_model is not None:
    temp_model_path = "_uploaded_model.pkcls"
    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    model_path = temp_model_path
elif model_choice_name is not None:
    model_path = os.path.join(MODEL_DIR, model_choice_name)

if model_path is None:
    st.info("กรุณาเลือกหรืออัปโหลดไฟล์โมเดล (.pkcls) ก่อน จึงจะเริ่มใช้งานได้")
    st.stop()


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path: str):
    """โหลดโมเดล Orange (.pkcls) ด้วย joblib และ cache ไว้ไม่ให้โหลดซ้ำทุกครั้ง"""
    return joblib.load(path)


try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

st.sidebar.success(f"โหลดโมเดล\n'{os.path.basename(model_path)}' สำเร็จ")

domain = model.domain  # Orange.data.Domain ที่ติดมากับตัวโมเดล (มาจากตอนฝึก)


# ----------------------------------------------------------------------------
# 3) อัปโหลดภาพ X-ray (แทนการกรอกค่าตัวเลขทีละช่อง เพราะ features เป็น
#    embedding vector ที่แปลงมาจากภาพ ไม่ใช่ค่าที่มนุษย์กรอกเองได้ตรง ๆ)
# ----------------------------------------------------------------------------
st.subheader("อัปโหลดภาพ X-ray")

if not IMAGE_EMBEDDER_AVAILABLE:
    st.error(
        "ไม่สามารถ import ไลบรารีแปลงภาพเป็น embedding ได้ "
        "กรุณาติดตั้ง Orange3-ImageAnalytics ก่อนใช้งาน (ดู requirements.txt)"
    )
    st.code(IMAGE_EMBEDDER_IMPORT_ERROR or "ไม่ทราบสาเหตุ (ไม่มีข้อความ error)")
    st.stop()

uploaded_image_file = st.file_uploader(
    "เลือกไฟล์ภาพ (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"],
)
if uploaded_image_file is not None:
    st.image(uploaded_image_file, caption="ภาพที่อัปโหลด", width=300)


# ----------------------------------------------------------------------------
# 4) ปุ่ม "ทำนายผล"
# ----------------------------------------------------------------------------
if st.button("ทำนายผล", type="primary"):
    if uploaded_image_file is None:
        st.warning("กรุณาอัปโหลดภาพ X-ray ก่อนเริ่มการทำนาย")
        st.stop()

    try:
        # เขียนไฟล์ภาพที่อัปโหลดลงดิสก์ชั่วคราว เพราะ ImageEmbedder ต้องการ
        # path ของไฟล์ภาพ ไม่ใช่ bytes ตรง ๆ
        suffix = os.path.splitext(uploaded_image_file.name)[1] or ".jpg"
        temp_image_path = f"_uploaded_image{suffix}"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_image_file.getbuffer())

        # ---------------------------------------------------------------
        # แปลงภาพเป็น embedding vector ด้วย SqueezeNet (ตัวเดียวกับตอนฝึก)
        # ---------------------------------------------------------------
        with st.spinner("กำลังแปลงภาพเป็นตัวเลข (embedding) ด้วย SqueezeNet..."):
            with ImageEmbedder(model="squeezenet") as embedder:
                embeddings = embedder([temp_image_path])

        embedding_vector = embeddings[0]
        if embedding_vector is None or len(embedding_vector) == 0:
            st.error("แปลงภาพไม่สำเร็จ (ไฟล์ภาพอาจเสียหายหรืออ่านไม่ได้)")
            st.stop()
        if len(embedding_vector) != len(domain.attributes):
            st.error(
                f"จำนวนมิติของ embedding ({len(embedding_vector)}) "
                f"ไม่ตรงกับที่โมเดลต้องการ ({len(domain.attributes)}) "
                "กรุณาตรวจสอบว่าโมเดลนี้ฝึกด้วย embedder ตัวเดียวกัน (SqueezeNet)"
            )
            st.stop()

        row = list(embedding_vector)
        X = np.array([row], dtype=float)

        # สร้างคอลัมน์คลาส (Y) เป็นค่า "ไม่ทราบค่า" (NaN) เพราะตอนทำนาย
        # เรายังไม่รู้คำตอบจริง แต่ Orange ต้องการให้ระบุจำนวนคอลัมน์คลาส
        # ให้ตรงกับ domain เสมอ (แม้ค่าจะเป็น NaN ก็ตาม) มิฉะนั้นจะเจอ
        # error "Invalid number of class columns"
        n_class_vars = len(domain.class_vars) if domain.class_vars else 0
        Y = np.full((X.shape[0], n_class_vars), np.nan) if n_class_vars else None

        # สร้าง Orange Table จาก domain เดิม (รับประกันว่าคอลัมน์/ลำดับตรงกับตอนฝึก)
        instance_table = Table.from_numpy(domain, X, Y)

        # ทำนายผล พร้อมความน่าจะเป็นของแต่ละคลาส
        pred_idx, probs = model(instance_table, ret=Orange.classification.Model.ValueProbs)

        class_var = domain.class_var
        predicted_label = class_var.values[int(pred_idx[0])]
        confidence = float(np.max(probs[0])) * 100

        # ----------------------------------------------------------------
        # 5) แสดงผลการทำนาย
        # ----------------------------------------------------------------
        st.success(f"ผลการทำนาย: **{predicted_label}** (ความมั่นใจ {confidence:.2f}%)")

        # แสดงความน่าจะเป็นของทุกคลาสแบบละเอียด เพื่อให้อ่านง่ายขึ้น
        st.write("ความน่าจะเป็นของแต่ละคลาส:")
        prob_dict = {
            class_var.values[i]: f"{p * 100:.2f}%"
            for i, p in enumerate(probs[0])
        }
        st.table(prob_dict)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างทำนายผล: {e}")
