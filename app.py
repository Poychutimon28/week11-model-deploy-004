# app.py
# ============================================================================
# โปรแกรมจำแนกโรค Covid-19 จากภาพ X-ray (Streamlit Web App)
# ----------------------------------------------------------------------------
# หมายเหตุสำคัญ (โปรดอ่านก่อนใช้งาน):
#   ไฟล์โมเดล .pkcls ทั้ง 3 ไฟล์ที่แนบมา (Logistic Regression, Neural Network,
#   Decision Tree) ถูกฝึกและบันทึกด้วยโปรแกรม "Orange Data Mining" ไม่ใช่
#   sklearn ล้วน ๆ ที่ pickle/joblib ธรรมดา ดังนั้นตัวโมเดลจะเป็น object
#   ชนิด Orange.base.Model ซึ่งมี "domain" (โครงสร้างคอลัมน์ที่ใช้ตอนฝึก)
#   ติดมากับตัวโมเดลเองอยู่แล้ว
#
#   ข้อดีคือ: เราไม่จำเป็นต้องรู้ล่วงหน้าว่ามีคอลัมน์อะไรบ้าง เพราะแอปนี้
#   จะ "อ่านโครงสร้างคอลัมน์ (domain) จากตัวโมเดลโดยอัตโนมัติ" แล้วสร้าง
#   ฟอร์มกรอกข้อมูลให้ตรงกับตอนฝึกเสมอ (ไม่ต้อง one-hot encode มือ เพราะ
#   Orange จัดการ categorical variable ให้เองผ่าน DiscreteVariable)
#
#   ข้อกำหนด: ต้องติดตั้งไลบรารี Orange3 ไว้ในเครื่อง/เซิร์ฟเวอร์ที่รันแอปนี้
#   ด้วย (ดูไฟล์ requirements.txt) ไม่เช่นนั้น joblib.load จะโหลดไฟล์ .pkcls
#   ไม่สำเร็จ (จะฟ้อง ModuleNotFoundError: No module named 'Orange')
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


# ----------------------------------------------------------------------------
# 2) หัวข้อของแอป
# ----------------------------------------------------------------------------
st.set_page_config(page_title="จำแนกโรค Covid-19 จากภาพ X-ray", page_icon="🩻")
st.title("โปรแกรมจำแนกโรค Covid-19 จากภาพ X-ray")
st.caption(
    "เลือกโมเดลที่ฝึกไว้ล่วงหน้า (.pkcls) จากนั้นกรอกค่าตัวแปรต้น (features) "
    "แล้วกดปุ่ม 'ทำนายผล' เพื่อดูผลการจำแนก"
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
# ----------------------------------------------------------------------------
# โฟลเดอร์ที่เก็บไฟล์โมเดล .pkcls ทั้งหมด (ต้องนำไฟล์ .pkcls ทั้ง 3 ไฟล์
# ไปวางไว้ในโฟลเดอร์ชื่อ "models" ที่ root ของ repo บน GitHub เดียวกับ app.py)
MODEL_DIR = "models"

# ค้นหาไฟล์ .pkcls ทั้งหมดในโฟลเดอร์ที่กำหนด
model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkcls")))

st.sidebar.header("⚙️ เลือกโมเดล")

if not model_files:
    st.sidebar.error(
        f"ไม่พบไฟล์ .pkcls ในโฟลเดอร์ '{MODEL_DIR}/' "
        "กรุณานำไฟล์โมเดล (.pkcls) ไปวางไว้ในโฟลเดอร์นี้บน GitHub repo "
        "แล้ว deploy ใหม่อีกครั้ง"
    )
    st.stop()

# ให้ผู้ใช้เลือกโมเดลจากรายการไฟล์ที่พบในโฟลเดอร์ models/
model_path = st.sidebar.selectbox(
    "เลือกไฟล์โมเดลที่ต้องการใช้ทำนาย",
    options=model_files,
    format_func=lambda p: os.path.basename(p),
)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path: str):
    """โหลดโมเดล Orange (.pkcls) ด้วย joblib และ cache ไว้ไม่ให้โหลดซ้ำทุกครั้ง"""
    return joblib.load(path)


try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

st.sidebar.success(f"โหลดโมเดล '{os.path.basename(model_path)}' สำเร็จ")


# ----------------------------------------------------------------------------
# 3) สร้างฟอร์มกรอกค่าตัวแปรต้น (features) โดยอ่านจาก domain ของโมเดลอัตโนมัติ
# ----------------------------------------------------------------------------
domain = model.domain  # Orange.data.Domain ที่ติดมากับตัวโมเดล (มาจากตอนฝึก)

st.subheader("กรอกค่าตัวแปรต้น (Features)")

user_values = {}  # เก็บค่าที่ผู้ใช้กรอก key = ชื่อ attribute, value = ค่าที่แปลงแล้ว (float)

for attr in domain.attributes:
    if isinstance(attr, ContinuousVariable):
        # ตัวแปรตัวเลขต่อเนื่อง -> ใช้ st.number_input
        val = st.number_input(
            label=attr.name,
            value=0.0,
            format="%.4f",
            key=f"num_{attr.name}",
        )
        user_values[attr.name] = float(val)

    elif isinstance(attr, DiscreteVariable):
        # ตัวแปรหมวดหมู่ (categorical) -> ใช้ st.selectbox โดยดึงรายการ
        # ค่าที่เป็นไปได้ (attr.values) มาจากตอนฝึกโมเดลโดยตรง
        # (Orange จะแปลงข้อความเป็นตัวเลขภายในให้เอง เทียบเท่ากับการทำ
        #  encoding/one-hot ตอนฝึก จึงไม่ต้อง one-hot ด้วยมืออีกครั้ง)
        selected_label = st.selectbox(
            label=attr.name,
            options=list(attr.values),
            key=f"sel_{attr.name}",
        )
        # แปลงข้อความที่เลือก -> ดัชนี (index) ตามลำดับใน attr.values
        # เพื่อให้ตรงรูปแบบตัวเลขที่ Orange ใช้ภายใน (เหมือนตอนฝึกโมเดล)
        user_values[attr.name] = float(attr.values.index(selected_label))

    else:
        st.warning(f"ไม่รองรับชนิดตัวแปร '{attr.name}' ({type(attr)}) โดยอัตโนมัติ")


# ----------------------------------------------------------------------------
# 4) ปุ่ม "ทำนายผล"
# ----------------------------------------------------------------------------
if st.button("ทำนายผล", type="primary"):
    try:
        if IS_IMAGE_MODEL:
            # ---------------------------------------------------------------
            # แปลงภาพที่อัปโหลดเป็น embedding vector ด้วย SqueezeNet ก่อน
            # ---------------------------------------------------------------
            if uploaded_image_file is None:
                st.warning("กรุณาอัปโหลดภาพ X-ray ก่อนกดทำนายผล")
                st.stop()

            # เขียนไฟล์ภาพที่อัปโหลดลงดิสก์ชั่วคราว เพราะ ImageEmbedder
            # ต้องการ path ของไฟล์ภาพ ไม่ใช่ bytes ตรง ๆ
            suffix = os.path.splitext(uploaded_image_file.name)[1] or ".jpg"
            temp_image_path = f"_uploaded_image{suffix}"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_image_file.getbuffer())

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
                    "กรุณาตรวจสอบว่าใช้ embedder ตัวเดียวกับตอนฝึกโมเดล (SqueezeNet)"
                )
                st.stop()

            row = list(embedding_vector)
        else:
            # จัดเรียงค่าตามลำดับ attribute เดียวกับตอน domain.attributes ถูกฝึกไว้
            row = [user_values[attr.name] for attr in domain.attributes]

        X = np.array([row], dtype=float)

        # สร้าง Orange Table จาก domain เดิม (รับประกันว่าคอลัมน์/ลำดับตรงกับตอนฝึก)
        instance_table = Table.from_numpy(domain, X)

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
