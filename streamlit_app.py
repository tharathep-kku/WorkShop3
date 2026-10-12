"""แอปประเมินความเสี่ยงงานยกเคลื่อนย้ายจากคลิปวิดีโอ — รันด้วย `streamlit run streamlit_app.py`"""
import hashlib, subprocess, tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

import altair as alt
import imageio_ffmpeg
import pandas as pd
import streamlit as st

import lifting as L

ROOT = Path(__file__).parent
TMP = Path(tempfile.gettempdir()) / "lifting_app"   # ponytail: ไม่ลบไฟล์เก่า · Streamlit Cloud ล้างเองเมื่อแอป restart
TMP.mkdir(exist_ok=True)

ZONE_TH = {"Above Shoulder": "เหนือหัวไหล่", "Waist-Shoulder": "เอวถึงหัวไหล่",
           "Knee-Waist": "หัวเข่าถึงเอว", "Below Knee": "ใต้เข่า"}
REACH_TH = {"Near": "ใกล้ (0–7″)", "Mild": "ปานกลาง (7–12″)", "Extended": "ระยะเอื้อม (>12″)"}
LEVEL_COLOR = {1: "#92D050", 2: "#FFFF66", 3: "#F6C344", 4: "#EE7651"}   # สีตามแบบประเมิน
LEVEL_RANGE = {1: "น้อยกว่า 50% RWL", 2: "50% – 75% RWL", 3: "มากกว่า 75% – 100% RWL", 4: "มากกว่า 100% RWL"}
PICK_COLOR, PLACE_COLOR = "#93C5FD", "#FCD34D"

st.set_page_config(page_title="แบบประเมินความเสี่ยงงานยก", page_icon="📦", layout="wide")


# ------------------------------------------------------------------ ส่วนช่วยแสดงผล
def mark(df, cells):
    # ระบายสีช่องของตาราง: cells = {(แถว, คอลัมน์): สี}
    css = pd.DataFrame("", index=df.index, columns=df.columns)
    for (row, col), color in cells.items():
        css.loc[row, col] = f"background-color: {color}; color: #111; font-weight: 700"
    return df.style.apply(lambda _: css, axis=None)


def station_text(s):
    return f"{ZONE_TH[s['zone']]} · {REACH_TH[s['reach_class']]}"


def timeline(results, series, y_title, levels=(), lifts=()):
    # กราฟตามเวลา: เส้นข้อมูล + เส้นประเกณฑ์ + แถบสีช่วงหยิบ→วางของแต่ละรอบ
    data = (results[["time_s", *series]].rename(columns=series)
            .melt("time_s", var_name="series", value_name="value").dropna())
    t_max = float(results["time_s"].max())
    layers = []
    if lifts:
        spans = pd.DataFrame([{"start": l["pick"]["time_s"], "end": l["place"]["time_s"], "label": f"#{l['lift']}"}
                              for l in lifts])
        layers.append(alt.Chart(spans).mark_rect(opacity=0.2, color="#f59e0b").encode(x="start:Q", x2="end:Q"))
        layers.append(alt.Chart(spans).mark_text(align="left", baseline="top", dx=3, dy=3, color="#b45309",
                                                 fontWeight="bold").encode(x="start:Q", y=alt.value(0), text="label:N"))
    layers.append(alt.Chart(data).mark_line(strokeWidth=2).encode(
        x=alt.X("time_s:Q", title="เวลา (วินาที)"), y=alt.Y("value:Q", title=y_title),
        color=alt.Color("series:N", legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("time_s:Q", title="วินาที", format=".1f"), alt.Tooltip("series:N", title="ชุดข้อมูล"),
                 alt.Tooltip("value:Q", title="ค่า", format=".1f")]))
    if levels:
        rules = pd.DataFrame(levels, columns=["value", "label"])
        layers.append(alt.Chart(rules).mark_rule(strokeDash=[4, 4], color="#6b7280").encode(y="value:Q"))
        layers.append(alt.Chart(rules).mark_text(align="right", dx=-4, dy=-7, color="#6b7280").encode(
            x=alt.datum(t_max), y="value:Q", text="label:N"))
    return alt.layer(*layers).properties(height=240)


@st.cache_data(show_spinner="กำลังคำนวณตำแหน่งมือ มุมบิด และรอบยก ...", max_entries=20)
def analyze(clip_key, video_path, height_cm, params, _keypoints):
    # คำนวณทุกเฟรมจาก keypoint ที่เก็บไว้ (ไม่รัน MediaPipe ซ้ำ) → เร็วพอให้ขยับ slider แล้วเห็นผลทันที
    p = L.Params(**params)
    cap, info = L.open_video(video_path)
    cap.release()
    results, calib = L.run_pipeline(video_path, None, height_cm, p, keypoints=_keypoints)
    if calib is None or "V_cm" not in results:
        return results, None, [], info
    levels = {k: float(calib[k + "_cm"]) for k in ("knee", "waist", "shoulder")}
    return results, levels, L.detect_lifts(results, calib, info["fps"], p), info


# ------------------------------------------------------------------ แถบด้านข้าง: ข้อมูลนำเข้า
with st.sidebar:
    st.header("1 · คลิปวิดีโอ")
    upload = st.file_uploader("อัปโหลดคลิปงานยก", type=["mp4", "mov", "avi", "m4v"])

    st.header("2 · ข้อมูลแบบประเมิน")
    company = st.text_input("บริษัท")
    job = st.text_input("ชื่องาน")
    assessor = st.text_input("ผู้ประเมิน")
    assess_date = st.date_input("วันที่", date.today(), format="DD/MM/YYYY")

    st.header("3 · ผู้ยกและงาน")
    height_cm = st.number_input("ส่วนสูงผู้ยก (ซม.)", 120.0, 220.0, 170.0, 1.0,
                                help="ใช้แปลงพิกเซลเป็นเซนติเมตร ใส่ส่วนสูงจริงของคนในคลิป")
    weight_kg = st.number_input("น้ำหนักวัตถุ (กก.) — ขั้นตอนที่ 1", 0.1, 100.0, 10.0, 0.5)
    hours = st.radio("ชั่วโมงทำงานยกต่อวัน — ขั้นตอนที่ 3", range(3), format_func=L.HOUR_COLS.__getitem__)

    with st.expander("ค่าปรับขั้นสูง"):
        default = L.Params()
        params = {
            "twist_warn_deg": st.slider("เกณฑ์บิดตัว (องศา)", 10.0, 60.0, default.twist_warn_deg, 1.0,
                                        help="แบบประเมินกำหนด 45° · กล้องตัวเดียววัดมุมบิดได้ต่ำกว่าจริง "
                                             "ถ้าคลิปบิดตัวชัดแต่ระบบไม่ขึ้น ให้ลดค่านี้"),
            "lift_min_move_cm": st.slider("ระยะยกขั้นต่ำที่นับเป็น 1 รอบ (ซม.)", 10.0, 50.0,
                                          default.lift_min_move_cm, 1.0),
            "waist_fraction": st.slider("ตำแหน่งเอว (สัดส่วนระหว่างสะโพกกับไหล่)", 0.20, 0.50,
                                        default.waist_fraction, 0.01),
            "camera_fov_deg": st.slider("มุมมองกล้อง FOV (องศา)", 50.0, 90.0, default.camera_fov_deg, 1.0,
                                        help="มือถือทั่วไป 65–75°"),
            "calib_seconds": st.slider("ช่วงยืนนิ่งต้นคลิป (วินาที)", 1.0, 3.0, default.calib_seconds, 0.1),
        }
        assert set(params) == set(asdict(default)), "slider ต้องครบทุกค่าใน Params"

# ------------------------------------------------------------------ หน้าหลัก
st.title("📦 แบบประเมินความเสี่ยงงานยกเคลื่อนย้าย")
st.caption("Lifting Handling Risk Assessment · วิเคราะห์ตำแหน่งมือ มุมบิดตัว และจำนวนรอบยกจากคลิปวิดีโอด้วย MediaPipe Pose")

if upload is None:
    st.info("อัปโหลดคลิปที่แถบด้านซ้ายเพื่อเริ่มประเมิน")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("วิธีถ่ายคลิปให้ผลแม่น")
        st.markdown(
            "1. ตั้งกล้องนิ่ง เห็นผู้ยก**เต็มตัวตั้งแต่หัวถึงเท้า**ตลอดคลิป\n"
            "2. ผู้ยก**ยืนตรง มือแนบลำตัว นิ่ง 2 วินาที**ก่อนเริ่ม (ระบบใช้ช่วงนี้วัดสัดส่วนร่างกาย)\n"
            "3. หลังวางของทุกครั้ง **ปล่อยมือกลับมาแนบลำตัว** ก่อนยกรอบถัดไป (ระบบใช้ท่านี้แบ่งรอบ)\n"
            "4. ถ่ายด้านข้างของผู้ยกจะวัดระยะยื่นมือได้ใกล้จริงที่สุด\n"
            "5. กรอกส่วนสูงจริงและน้ำหนักวัตถุที่แถบด้านซ้าย")
        st.subheader("ระบบคำนวณอะไรให้")
        st.markdown(
            "- **ขั้นตอนที่ 2** น้ำหนักจากตาราง ณ ตำแหน่งมือตอนหยิบและตอนวาง\n"
            "- **ขั้นตอนที่ 3** ตัวคูณความถี่จากจำนวนรอบที่นับได้ในคลิป\n"
            "- **ขั้นตอนที่ 4** ตัวคูณบิดลำตัวจากมุมบิดสูงสุดขณะถือของ\n"
            "- **ขั้นตอนที่ 5–7** ค่า RWL · LH Index · ระดับความเสี่ยง 1–4")
    with right:
        st.image(str(ROOT / "docs" / "lifting_risk_form.png"), caption="แบบประเมินต้นฉบับ")
    st.stop()

# ---- ① สกัด keypoint (ช้า ทำครั้งเดียวต่อคลิป เก็บไว้ใน session)
data = upload.getvalue()
clip_key = hashlib.md5(data).hexdigest()
video_path = TMP / f"{clip_key}{Path(upload.name).suffix.lower()}"
if not video_path.is_file():
    video_path.write_bytes(data)
if st.session_state.get("clip_key") != clip_key:
    bar = st.progress(0.0, text="ตรวจจับท่าทางด้วย MediaPipe ...")
    st.session_state.keypoints = L.extract_video_keypoints(
        video_path, lambda f: bar.progress(f, text=f"ตรวจจับท่าทางด้วย MediaPipe ... {f:.0%}"))
    st.session_state.clip_key = clip_key
    st.session_state.pop("video", None)
    bar.empty()
keypoints = st.session_state.keypoints
if len(keypoints["xy"]) == 0:
    st.error("อ่านวิดีโอไม่ได้ ลองแปลงไฟล์เป็น .mp4 (H.264) แล้วอัปโหลดใหม่")
    st.stop()

# ---- ② คำนวณทุกเฟรม + นับรอบ
try:
    results, levels, lifts, info = analyze(clip_key, str(video_path), height_cm, params, keypoints)
except Exception as error:   # คลิปที่ไม่มีคนยืนนิ่งช่วงต้นทำให้ calibrate ล้ม → บอกผู้ใช้แทนการขึ้นหน้า error
    st.error("ปรับเทียบจากช่วงต้นคลิปไม่สำเร็จ ตรวจว่าผู้ยกยืนตรงนิ่งเห็นเต็มตัวในช่วงแรกของคลิป")
    with st.expander("รายละเอียดข้อผิดพลาด"):
        st.exception(error)
    st.stop()
if levels is None:
    st.error(f"คลิปสั้นเกินไป หรือตรวจไม่พบผู้ยกหลังช่วงยืนนิ่ง {params['calib_seconds']:.1f} วินาทีแรก")
    st.stop()

valid = results.dropna(subset=["V_cm"])
active_min = (results["time_s"].iloc[-1] - valid["time_s"].iloc[0]) / 60
rate = len(lifts) / active_min if active_min > 0 else 0.0

# ตัวคูณความถี่: ค่าเริ่มต้นจากคลิป · ผู้ใช้เลือกแถวเองได้ในขั้นตอนที่ 3 (อ่านค่าจาก session เพราะแท็บอื่นต้องใช้ก่อนวาด widget)
freq = st.session_state.get(f"freq_{clip_key}", L.freq_row(rate))
freq_mult = L.FREQ_MULTIPLIER[freq][hours]
rows = [{**l, **L.assess(weight_kg, l["limit_kg"], freq_mult, l["twist_max_deg"], params["twist_warn_deg"])}
        for l in lifts]

tab_form, tab_video, tab_detail = st.tabs(["📋 ผลประเมิน", "🎬 วิดีโอผลลัพธ์", "📈 รายรอบและกราฟ"])

# ------------------------------------------------------------------ แท็บ 2: วิดีโอผลลัพธ์
with tab_video:
    signature = (clip_key, height_cm, tuple(params.values()))
    video = st.session_state.get("video")
    if video and video["signature"] != signature:
        st.info("ค่าที่ตั้งเปลี่ยนไปจากตอนสร้างวิดีโอ กดสร้างใหม่เพื่อให้ตรงกับผลประเมินล่าสุด")
    if st.button("🎬 สร้างวิดีโอผลลัพธ์", type="primary",
                 help="วาดโครงกระดูก ตารางน้ำหนัก และตัวนับรอบลงบนคลิป ใช้เวลาราว 1–2 เท่าของความยาวคลิป"):
        raw, final = TMP / f"{clip_key}_raw.mp4", TMP / f"{clip_key}_result.mp4"
        bar = st.progress(0.0, text="กำลังวาดผลลงบนวิดีโอ ...")
        L.run_pipeline(video_path, raw, height_cm, L.Params(**params), keypoints=keypoints, lifts=lifts,
                       progress=lambda f: bar.progress(f, text=f"กำลังวาดผลลงบนวิดีโอ ... {f:.0%}"))
        bar.progress(1.0, text="กำลังแปลงเป็น H.264 ...")   # mp4v ของ OpenCV เล่นในเบราว์เซอร์ไม่ได้
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-preset", "veryfast", "-pix_fmt", "yuv420p", "-crf", "26", str(final)], check=True)
        raw.unlink()
        bar.empty()
        video = st.session_state.video = {"signature": signature, "path": str(final)}
    if video:
        st.video(video["path"])
        st.download_button("⬇️ ดาวน์โหลดวิดีโอ (.mp4)", Path(video["path"]).read_bytes(),
                           f"{Path(upload.name).stem}_result.mp4", "video/mp4")
    else:
        st.caption("ยังไม่ได้สร้างวิดีโอ · ผลประเมินในแท็บแรกไม่ต้องรอวิดีโอ")

# ------------------------------------------------------------------ แท็บ 3: รายรอบและกราฟ
with tab_detail:
    st.subheader("สรุปรายรอบยก")
    lift_table = pd.DataFrame([{
        "รอบ": r["lift"],
        "หยิบ (วินาที)": round(r["pick"]["time_s"], 1), "หยิบจาก": station_text(r["pick"]),
        "กก. ตอนหยิบ": r["pick"]["limit_kg"],
        "วาง (วินาที)": round(r["place"]["time_s"], 1), "วางที่": station_text(r["place"]),
        "กก. ตอนวาง": r["place"]["limit_kg"],
        "ขั้นตอนที่ 2 (กก.)": r["limit_kg"],
        "บิดสูงสุด (°)": round(r["twist_max_deg"]), "ตัวคูณบิด": r["twist_multiplier"],
        "ระยะเอื้อมสูงสุด (ซม.)": round(r["reach_max_cm"]),
        "RWL (กก.)": round(r["rwl_kg"], 2),
        "LH Index (%)": None if r["lh_index"] == float("inf") else round(r["lh_index"]),
        "ระดับ": r["level"],
    } for r in rows])
    if rows:
        st.dataframe(lift_table.style.apply(
            lambda row: [f"background-color: {LEVEL_COLOR[row['ระดับ']]}; color: #111" if col == "ระดับ" else ""
                         for col in row.index], axis=1).format(precision=2), hide_index=True)
    else:
        st.caption("ไม่พบรอบยก — ดูเส้นความสูงมือด้านล่างเพื่อหาสาเหตุ")

    st.subheader("เส้นเวลา")
    st.caption("แถบสีส้ม = ช่วงถือของ (หยิบ → วาง) ของแต่ละรอบ · เส้นประ = เกณฑ์ของตาราง")
    st.altair_chart(timeline(results, {"V_cm": "ความสูงมือ V"}, "ความสูงจากพื้น (ซม.)",
                             [(levels["knee"], "เข่า"), (levels["waist"], "เอว"), (levels["shoulder"], "ไหล่")], lifts))
    st.altair_chart(timeline(results, {"H_cm": "ระยะยื่น H (จากข้อเท้า)", "reach_cm": "ระยะเอื้อม G13 (จากลำตัว)"},
                             "ระยะ (ซม.)", [(L.REACH_LIMITS_CM[0], "7″"), (L.REACH_LIMITS_CM[1], "12″")], lifts))
    st.altair_chart(timeline(results, {"twist_deg": "มุมบิดตัว", "orientation_deg": "มุมหันเทียบกล้อง"},
                             "มุม (องศา)", [(params["twist_warn_deg"], "เกณฑ์บิดตัว")], lifts))
    st.caption(f"ระดับที่วัดได้ตอนยืนนิ่ง: เข่า {levels['knee']:.0f} ซม. · เอว {levels['waist']:.0f} ซม. "
               f"· ไหล่ {levels['shoulder']:.0f} ซม. · มุมหันเทียบกล้องควรใกล้ 90° (ถ่ายด้านข้าง)")

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ ตารางรายรอบ (.csv)", lift_table.to_csv(index=False).encode("utf-8-sig"),
                       f"{Path(upload.name).stem}_lifts.csv", "text/csv", disabled=not rows)
    c2.download_button("⬇️ ค่ารายเฟรม (.csv)", results.to_csv(index=False).encode("utf-8-sig"),
                       f"{Path(upload.name).stem}_frames.csv", "text/csv")

# ------------------------------------------------------------------ แท็บ 1: แบบประเมิน 7 ขั้นตอน (วาดท้ายสุดเพราะมี st.stop)
with tab_form:
    if not lifts:
        st.warning("ไม่พบรอบยกในคลิปนี้ — ตรวจว่าผู้ยกปล่อยมือกลับมาแนบลำตัวหลังวางของ "
                   "หรือลด “ระยะยกขั้นต่ำที่นับเป็น 1 รอบ” ในค่าปรับขั้นสูง")
        st.stop()                                # แท็บอื่นวาดไปแล้วด้านบน

    summary = st.container()                     # เติมหลังรู้ตัวคูณความถี่ (เลือกในขั้นตอนที่ 3 ด้านล่าง)
    st.divider()
    left, right = st.columns(2, gap="large")

    with right:
        with st.container(border=True):
            st.markdown("#### ขั้นตอนที่ 3 · จำนวนรอบของการยก")
            st.caption(f"ระบบนับได้ **{len(lifts)} รอบ** ใน {active_min * 60:.0f} วินาที ≈ **{rate:.1f} รอบ/นาที** "
                       "· ถ้าความถี่ของงานจริงต่างจากในคลิป ให้เลือกแถวเอง")
            st.selectbox("ความถี่การยก", range(len(L.FREQ_ROWS)), index=L.freq_row(rate),
                         format_func=L.FREQ_ROWS.__getitem__, key=f"freq_{clip_key}")
            freq_table = pd.DataFrame(L.FREQ_MULTIPLIER, index=L.FREQ_ROWS, columns=L.HOUR_COLS)
            st.table(mark(freq_table, {(L.FREQ_ROWS[freq], L.HOUR_COLS[hours]): PLACE_COLOR}).format("{:.2f}"))
            st.markdown(f"ตัวคูณความถี่ = **{freq_mult:.2f}**")

    worst = min(rows, key=lambda r: r["rwl_kg"])    # สรุปด้วยรอบที่ RWL ต่ำสุด (แย่ที่สุด)
    index_text = "เกิน 100%" if worst["lh_index"] == float("inf") else f"{worst['lh_index']:.0f}%"

    with summary:
        header = " · ".join(text for text in (company and f"บริษัท {company}", job and f"งาน {job}",
                                               assessor and f"ผู้ประเมิน {assessor}",
                                               assess_date.strftime("%d/%m/%Y")) if text)
        st.caption(header)
        st.markdown(
            f"<div style='background:{LEVEL_COLOR[worst['level']]};color:#111;padding:18px 22px;border-radius:10px'>"
            f"<div style='font-size:1.6rem;font-weight:700'>ความเสี่ยงระดับ {worst['level']} · LH Index {index_text}</div>"
            f"<div style='font-size:1.05rem'>{worst['meaning']}</div></div>", unsafe_allow_html=True)
        st.write("")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("จำนวนรอบยกที่นับได้", f"{len(lifts)} รอบ")
        m2.metric("ค่าขีดจำกัดการยก (RWL)", f"{worst['rwl_kg']:.1f} กก.")
        m3.metric("น้ำหนักวัตถุ", f"{weight_kg:g} กก.")
        m4.metric("บิดตัวสูงสุดขณะยก", f"{max(r['twist_max_deg'] for r in rows):.0f}°")
        if len(rows) > 1:
            st.caption(f"สรุปจากรอบที่ {worst['lift']} ซึ่งมีค่า RWL ต่ำสุดจาก {len(rows)} รอบ "
                       "· ดูทุกรอบได้ที่แท็บ “รายรอบและกราฟ”")

    with left:
        with st.container(border=True):
            st.markdown("#### ขั้นตอนที่ 1 · น้ำหนักวัตถุสิ่งของที่ใช้ยก")
            st.markdown(f"น้ำหนักยก = **{weight_kg:g} กิโลกรัม**")
        with st.container(border=True):
            st.markdown("#### ขั้นตอนที่ 2 · ตำแหน่งของมือขณะยกและวางวัตถุ")
            pick, place = worst["pick"], worst["place"]
            weight_table = pd.DataFrame(L.WEIGHT_LIMITS_KG, index=[ZONE_TH[z] for z in L.ROW_NAMES],
                                        columns=[REACH_TH[c] for c in L.COL_NAMES])
            cells = {(ZONE_TH[pick["zone"]], REACH_TH[pick["reach_class"]]): PICK_COLOR,
                     (ZONE_TH[place["zone"]], REACH_TH[place["reach_class"]]): PLACE_COLOR}
            st.table(mark(weight_table, cells))
            st.markdown(
                f"<span style='background:{PICK_COLOR};color:#111;padding:1px 8px;border-radius:4px'>หยิบ</span> "
                f"{station_text(pick)} → **{pick['limit_kg']} กก.** (วินาทีที่ {pick['time_s']:.1f})<br>"
                f"<span style='background:{PLACE_COLOR};color:#111;padding:1px 8px;border-radius:4px'>วาง</span> "
                f"{station_text(place)} → **{place['limit_kg']} กก.** (วินาทีที่ {place['time_s']:.1f})",
                unsafe_allow_html=True)
            st.markdown(f"ใช้ค่าที่น้อยกว่า = **{worst['limit_kg']} กก.**")

    with right:
        with st.container(border=True):
            st.markdown("#### ขั้นตอนที่ 4 · การบิด/หมุนลำตัวขณะยก")
            twisted = worst["twist_multiplier"] < 1
            st.markdown(f"มุมบิดสูงสุดขณะถือของ = **{worst['twist_max_deg']:.0f}°** "
                        f"({'ถึง' if twisted else 'ไม่ถึง'}เกณฑ์ {params['twist_warn_deg']:.0f}°) "
                        f"→ ตัวคูณ = **{worst['twist_multiplier']:.2f}**")

    with st.container(border=True):
        st.markdown("#### ขั้นตอนที่ 5 · ค่าขีดจำกัดของการยกย้าย (RWL)")
        st.markdown(f"### {worst['limit_kg']} กก. × {freq_mult:.2f} × {worst['twist_multiplier']:.2f} "
                    f"= {worst['rwl_kg']:.2f} กก.")
        st.caption("ขั้นตอนที่ 2 × ขั้นตอนที่ 3 × ขั้นตอนที่ 4")
    with st.container(border=True):
        st.markdown("#### ขั้นตอนที่ 6 · ดัชนีงานยกเคลื่อนย้าย (LH Index)")
        if worst["rwl_kg"] > 0:
            st.markdown(f"### {weight_kg:g} ÷ {worst['rwl_kg']:.2f} × 100 = {worst['lh_index']:.0f}%")
        else:
            st.markdown("### RWL = 0 → ไม่ควรยกที่ความถี่นี้")
        st.caption("น้ำหนักขั้นตอนที่ 1 ÷ RWL × 100")
    with st.container(border=True):
        st.markdown("#### ขั้นตอนที่ 7 · สรุปผลระดับความเสี่ยง")
        risk_table = pd.DataFrame([{"ความเสี่ยง": f"ระดับ {lv}", "LH Index": LEVEL_RANGE[lv], "ความหมาย": text}
                                   for lv, text in L.RISK_LEVELS]).set_index("ความเสี่ยง")
        st.table(mark(risk_table, {(f"ระดับ {worst['level']}", col): LEVEL_COLOR[worst["level"]]
                                   for col in risk_table.columns}))

    report = pd.DataFrame([
        ("บริษัท", company), ("ชื่องาน", job), ("ผู้ประเมิน", assessor), ("วันที่", assess_date.strftime("%d/%m/%Y")),
        ("คลิป", upload.name), ("ส่วนสูงผู้ยก (ซม.)", height_cm),
        ("ขั้นตอนที่ 1 น้ำหนักวัตถุ (กก.)", weight_kg),
        ("ขั้นตอนที่ 2 ตำแหน่งหยิบ", station_text(pick)), ("ขั้นตอนที่ 2 ตำแหน่งวาง", station_text(place)),
        ("ขั้นตอนที่ 2 น้ำหนักจากตาราง (กก.)", worst["limit_kg"]),
        ("ขั้นตอนที่ 3 ความถี่", f"{L.FREQ_ROWS[freq]} · {L.HOUR_COLS[hours]}"), ("ขั้นตอนที่ 3 ตัวคูณ", freq_mult),
        ("ขั้นตอนที่ 4 มุมบิดสูงสุด (องศา)", round(worst["twist_max_deg"], 1)),
        ("ขั้นตอนที่ 4 ตัวคูณ", worst["twist_multiplier"]),
        ("ขั้นตอนที่ 5 RWL (กก.)", round(worst["rwl_kg"], 2)),
        ("ขั้นตอนที่ 6 LH Index (%)", index_text), ("ขั้นตอนที่ 7 ระดับความเสี่ยง", worst["level"]),
        ("ความหมาย", worst["meaning"]), ("จำนวนรอบที่นับได้", len(lifts)),
    ], columns=["รายการ", "ค่า"])
    st.download_button("⬇️ ดาวน์โหลดผลประเมิน (.csv)", report.to_csv(index=False).encode("utf-8-sig"),
                       f"{Path(upload.name).stem}_assessment.csv", "text/csv")
