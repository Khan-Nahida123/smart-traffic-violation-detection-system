"""
Streamlit UI for Smart Traffic Violation Detection System
"""

import streamlit as st
import requests


# -------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------

st.set_page_config(
    page_title="Smart Traffic Violation Detection System",
    layout="wide"
)


# -------------------------------------------------------------
# Dropdown interaction styling
# -------------------------------------------------------------

st.markdown(
    """
    <style>
    div[data-baseweb="select"] * {
        cursor: pointer !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------------------
# Title and subtitle
# -------------------------------------------------------------

st.header("Smart Traffic Violation Detection System")

st.caption(
    "Upload a vehicle image to detect the license plate and generate a traffic violation notice."
)

API_BASE = "http://127.0.0.1:8000"

st.divider()


# -------------------------------------------------------------
# Layout
# -------------------------------------------------------------

left_col, right_col = st.columns(2)


# -------------------------------------------------------------
# Violation settings
# -------------------------------------------------------------

with left_col:

    st.subheader("Violation Settings")

    violation = st.selectbox(
        "Violation Type",
        [
            "No Violation",
            "No Helmet",
            "Signal Jump",
            "Wrong Parking",
            "No Seatbelt",
            "Overspeeding",
        ],
    )

    uploaded_file = st.file_uploader(
        "Upload Vehicle Image",
        type=["jpg", "jpeg", "png"]
    )

    run_detection = st.button("Detect Plate", use_container_width=True)


# -------------------------------------------------------------
# Vehicle image preview
# -------------------------------------------------------------

with right_col:

    st.subheader("Vehicle Image")

    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True)

st.divider()


# -------------------------------------------------------------
# Processing
# -------------------------------------------------------------

if uploaded_file is not None and run_detection:

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }

    data = {
        "violation_type": violation
    }

    try:

        with st.spinner("Processing image..."):

            response = requests.post(
                f"{API_BASE}/anpr",
                files=files,
                data=data,
                timeout=120
            )

            result = response.json()

        # ---------------------------------------------------------
        # Detection Result Card
        # ---------------------------------------------------------

        with st.container(border=True):

            st.subheader("Detection Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.markdown("**Plate Number**")
                st.markdown("**Violation**")
                st.markdown("**Fine Amount**")

            with result_col2:
                st.write(result.get("plate"))
                st.write(result.get("violation"))
                st.write(f"₹{result.get('fine')}")

        if result.get("email_sent"):
            st.info("Email Notification Sent")
        else:
            st.warning("Email not sent")

    except Exception as e:
        st.error(f"API error: {e}")

st.divider()


# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------

st.caption("Powered by YOLOv8 | EasyOCR | FastAPI | Streamlit")