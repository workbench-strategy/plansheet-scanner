import streamlit as st
import logging
# Suppress 'missing ScriptRunContext' warnings from Streamlit
logging.getLogger('streamlit').setLevel(logging.ERROR)
from streamlit.elements.lib.image_utils import image_to_url
import streamlit.elements.image as st_image
# Monkey-patch image_to_url into streamlit.elements.image so st_canvas can use it
st_image.image_to_url = image_to_url  # type: ignore[attr-defined]
from streamlit_drawable_canvas import st_canvas
import fitz  # PyMuPDF
from PIL import Image
# Disable PIL decompression bomb limit for large page images (use cautiously)
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import os
import io
import base64  # add base64 import
import cv2
from glob import glob

# Constants
OUTPUT_DIR = "templates"
os.makedirs(os.path.join(OUTPUT_DIR, "existing"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "proposed"), exist_ok=True)

st.set_page_config(layout="wide")
st.title("Legend Symbol Extractor")

# --- Session State Initialization ---
if 'drawing_mode' not in st.session_state:
    st.session_state.drawing_mode = "rect"
if 'stroke_color' not in st.session_state:
    st.session_state.stroke_color = "red"
if 'bg_image' not in st.session_state:
    st.session_state.bg_image = None
if 'page_number' not in st.session_state:
    st.session_state.page_number = 0
if 'pdf_doc' not in st.session_state:
    st.session_state.pdf_doc = None
if 'image_to_display' not in st.session_state:
    st.session_state.image_to_display = None
if 'initial_drawing' not in st.session_state:
    # Store canvas drawing JSON; default empty dict to avoid stacking shapes
    st.session_state.initial_drawing = {}
if 'overwrite' not in st.session_state:
    st.session_state.overwrite = False


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        # Store PDF document in session state to avoid reloading on every interaction
        if st.session_state.pdf_doc is None or st.session_state.pdf_doc.name != uploaded_file.name:
            st.session_state.pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            st.session_state.page_number = 0 # Reset page number for new PDF
            st.session_state.image_to_display = None # Clear previous image

        pdf_doc = st.session_state.pdf_doc
        
        # Ensure page_number is valid
        if not 0 <= st.session_state.page_number < pdf_doc.page_count:
            st.session_state.page_number = 0

        # Allow user to choose DPI for higher-resolution symbol cropping
        dpi = st.sidebar.slider(
            "Image DPI (higher = better resolution)",
            min_value=50,
            max_value=400,
            value=200,
            step=50,
            help="Increase this for higher-res crops. Max 400 to avoid OOM"
        )
        page_number = st.number_input(
            f"Enter page number (0 to {pdf_doc.page_count - 1})", 
            min_value=0, 
            max_value=pdf_doc.page_count - 1, 
            value=st.session_state.page_number,
            key="page_num_input" 
        )
        st.session_state.page_number = page_number # Update session state from input

        # Load and display the page if it hasn't been loaded or page number changed
        if st.session_state.image_to_display is None or st.session_state.current_page_displayed != page_number or st.session_state.current_dpi != dpi:
            # Load and render page at the chosen DPI
            page = pdf_doc.load_page(st.session_state.page_number)
            pix = page.get_pixmap(dpi=dpi)  # type: ignore[attr-defined]
            # Save current DPI to session to re-render when DPI changes
            st.session_state.current_dpi = dpi
            img_bytes = pix.tobytes("png")
            st.session_state.image_to_display = Image.open(io.BytesIO(img_bytes))
            st.session_state.current_page_displayed = st.session_state.page_number


        if st.session_state.image_to_display:
            # Use displayed PIL image directly as background_image
            background_image = st.session_state.image_to_display
             
            img_display_width = st.session_state.image_to_display.width
            img_display_height = st.session_state.image_to_display.height
            
            # --- Symbol extraction form ---
            st.sidebar.header("Symbol Details")
            symbol_name = st.sidebar.text_input("Symbol Name (e.g., CCTV)")
            category = st.sidebar.selectbox("Category", ["existing", "proposed"])
            # Buttons: Clear current selection or Save symbol
            clear_button = st.sidebar.button("Clear Selection")
            save_button = st.sidebar.button("Save Selected Symbol")

            st.sidebar.info("Draw a rectangle around the symbol on the image.")
            
            # Reset drawing if Clear Selection clicked
            if clear_button:
                # Clear canvas drawing state and rerun to reset widget
                st.session_state.initial_drawing = {}
                st.sidebar.info("Selection cleared. Draw a new rectangle.")
                st.rerun()

             # --- Zoom & Tool Mode ---
            zoom_pct = st.sidebar.slider(
                "Zoom (%)", min_value=50, max_value=150, value=100, step=10,
                help="Zoom image for easier selection (50–150%)"
            )
            mode = st.sidebar.radio(
                "Tool", ["Draw Box", "Pan/Select"], index=0,
                help="Choose drawing rectangle or panning mode"
            )
            # Determine drawing mode
            draw_mode = "rect" if mode == "Draw Box" else "transform"
            st.session_state.drawing_mode = draw_mode
            # Resize image for zoom
            if zoom_pct != 100:
                zw = int(img_display_width * zoom_pct / 100)
                zh = int(img_display_height * zoom_pct / 100)
                display_img = background_image.resize((zw, zh))
            else:
                display_img = background_image
            # Use display_img in canvas
             
            # --- Drawable Canvas ---
            canvas_width = img_display_width
            canvas_height = img_display_height

            # Draw or Pan mode with optional initial drawing JSON
            canvas_result = st_canvas(
                 fill_color="rgba(255, 165, 0, 0.3)",  # Orange fill
                 stroke_width=2,
                 stroke_color=st.session_state.stroke_color,
                 background_image=display_img,  # type: ignore[arg-type]
                 initial_drawing=st.session_state.initial_drawing or {},
                 update_streamlit=True, # Realtime update
                 height=canvas_height * zoom_pct // 100,
                 width=canvas_width * zoom_pct // 100,
                 drawing_mode=st.session_state.drawing_mode,
                 key="canvas",
             )
            # Save latest drawing state for next render (enables panning/selection toggle)
            st.session_state.initial_drawing = canvas_result.json_data or {}

            if save_button and canvas_result.json_data:
                 objects = canvas_result.json_data["objects"]
                 if objects and objects[0]["type"] == "rect":
                     rect = objects[-1]  # Get the last drawn rectangle
                     left = int(rect["left"] * 100 / zoom_pct)
                     top = int(rect["top"] * 100 / zoom_pct)
                     width = int(rect["width"] * 100 / zoom_pct)
                     height = int(rect["height"] * 100 / zoom_pct)
                     
                     if not symbol_name.strip():
                         st.sidebar.error("Symbol name cannot be empty.")
                     elif width > 0 and height > 0:
                         # Crop from the original high-res image if needed, or from displayed one
                         # For now, cropping from the displayed image (which has DPI 200)
                         # Ensure coordinates are within image bounds
                         img_to_crop = st.session_state.image_to_display.copy()
                         
                         # Convert to RGB if it's RGBA (PIL requirement for some saves)
                         if img_to_crop.mode == 'RGBA':
                             img_to_crop = img_to_crop.convert('RGB')
                         
                         cropped_symbol = img_to_crop.crop((left, top, left + width, top + height))
                         
                         category_folder = os.path.join(OUTPUT_DIR, category.lower())
                         save_path = os.path.join(category_folder, f"{symbol_name}.png")
                         # confirm overwrite
                         overwrite_confirm = st.sidebar.checkbox(
                             "Overwrite existing file?", key="overwrite"
                         )
                         if os.path.exists(save_path) and not overwrite_confirm:
                             st.sidebar.warning(
                                 "Check 'Overwrite existing file?' to replace the template, or use a different name."
                         )
                         else:
                             try:
                                 cropped_symbol.save(save_path)
                                 st.sidebar.success(
                                     f"Symbol '{symbol_name}' saved to {save_path}"
                                 )
                                 # clear the rectangle and reset overwrite flag
                                 st.session_state.initial_drawing = {}
                                 st.session_state.overwrite = False
                             except Exception as e:
                                 st.sidebar.error(f"Error saving symbol: {e}")
                     else:
                         st.sidebar.warning("No valid rectangle drawn or selected.")
                 else:
                     st.sidebar.warning("Please draw a rectangle first.")
        else:
            st.info("Upload a PDF and select a page to begin.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.pdf_doc = None # Reset on error
        st.session_state.image_to_display = None

else:
    st.info("Upload a PDF file to get started.")

# --- Plan File Processing ---
st.sidebar.markdown("---")
st.sidebar.header("Plan File Processing")
plan_file = st.sidebar.file_uploader("Choose Plan‐Sheets PDF", type="pdf", key="plan_pdf")
rot_step = st.sidebar.select_slider(
    "Rotation Step",
    options=[0, 15, 30, 45, 90],
    value=0,
    help="Rotate each symbol template by ±this step when searching"
)
threshold = st.sidebar.slider(
    "Match Threshold",
    min_value=0.50,
    max_value=0.99,
    value=0.82,
    step=0.01,
    help="Lower → more (but fuzzier) matches; Higher → stricter"
)

template_pngs_exist = bool(glob("templates/*/*.png"))

if plan_file is not None and template_pngs_exist:
    # load plan PDF into PIL or OpenCV images, one per page
    doc = fitz.open(stream=plan_file.read(), filetype="pdf")
    for page in doc:
        pix = page.get_pixmap(dpi=200)  # type: ignore[attr-defined]
        img = cv2.cvtColor(
            np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n),
            cv2.COLOR_BGRA2BGR
        )
        results = []
        for tpl_path in glob("templates/*/*.png"):
            tpl = cv2.imread(tpl_path, cv2.IMREAD_UNCHANGED)
            for angle in range(0, 360, rot_step or 360):
                M = cv2.getRotationMatrix2D((tpl.shape[1]//2, tpl.shape[0]//2), angle, 1.0)
                tpl_rot = cv2.warpAffine(tpl, M, (tpl.shape[1], tpl.shape[0]))
                res = cv2.matchTemplate(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                        cv2.cvtColor(tpl_rot, cv2.COLOR_BGR2GRAY),
                                        cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):
                    results.append((tpl_path, angle, pt, float(res[pt[::-1]])))
        # Display or export these `results` however you like… e.g. overlay on the page,
        # list as a DataFrame, or hand off to your KMZ generator.

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for plan sheet scanning.")

# To run this app:
# 1. Save this code as streamlit_legend_extractor.py
# 2. Open your terminal in the same directory.
# 3. Run: streamlit run streamlit_legend_extractor.py
