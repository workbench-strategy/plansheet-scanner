import streamlit as st
from streamlit_drawable_canvas import st_canvas
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import os
import io

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

        page_number = st.number_input(
            f"Enter page number (0 to {pdf_doc.page_count - 1})", 
            min_value=0, 
            max_value=pdf_doc.page_count - 1, 
            value=st.session_state.page_number,
            key="page_num_input" 
        )
        st.session_state.page_number = page_number # Update session state from input

        # Load and display the page if it hasn't been loaded or page number changed
        if st.session_state.image_to_display is None or st.session_state.current_page_displayed != page_number:
            page = pdf_doc.load_page(st.session_state.page_number)
            pix = page.get_pixmap(dpi=200)  # Lower DPI for faster web display
            img_bytes = pix.tobytes("png")
            st.session_state.image_to_display = Image.open(io.BytesIO(img_bytes))
            st.session_state.current_page_displayed = st.session_state.page_number


        if st.session_state.image_to_display:
            img_display_width = st.session_state.image_to_display.width
            img_display_height = st.session_state.image_to_display.height
            
            # --- Symbol extraction form ---
            st.sidebar.header("Symbol Details")
            symbol_name = st.sidebar.text_input("Symbol Name (e.g., CCTV)")
            category = st.sidebar.selectbox("Category", ["existing", "proposed"])
            save_button = st.sidebar.button("Save Selected Symbol")

            st.sidebar.info("Draw a rectangle around the symbol on the image.")
            
            # --- Drawable Canvas ---
            # Adjust canvas size dynamically or set a max width/height
            # For simplicity, using image dimensions. Consider constraining for very large images.
            canvas_width = img_display_width
            canvas_height = img_display_height

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Orange fill with transparency
                stroke_width=2,
                stroke_color=st.session_state.stroke_color,
                background_image=st.session_state.image_to_display,
                update_streamlit=True, # Realtime update
                height=canvas_height,
                width=canvas_width,
                drawing_mode=st.session_state.drawing_mode,
                key="canvas",
            )

            if save_button and canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if objects and objects[0]["type"] == "rect": # Assuming the last drawn object is the one to save
                    rect = objects[-1] # Get the last drawn rectangle
                    left, top, width, height = int(rect["left"]), int(rect["top"]), int(rect["width"]), int(rect["height"])
                    
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
                        
                        try:
                            cropped_symbol.save(save_path)
                            st.sidebar.success(f"Symbol '{symbol_name}' saved to {save_path}")
                            # Optionally clear the drawing or the last object
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

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for plan sheet scanning.")

# To run this app:
# 1. Save this code as streamlit_legend_extractor.py
# 2. Open your terminal in the same directory.
# 3. Run: streamlit run streamlit_legend_extractor.py
