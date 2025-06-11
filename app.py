import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Prediksi Stunting - WHO Standards",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stunted {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .not-stunted {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .severely-stunted {
        background-color: #fce4ec;
        border-left: 5px solid #e91e63;
    }
</style>
""", unsafe_allow_html=True)

# WHO Height-for-age reference function
def get_height_for_age_reference(age_months, sex):
    """
    Returns median (M), standard deviation (SD) for height-for-age based on WHO standards
    """
    references = {
        0: (49.9, 1.9, 49.1, 1.9),
        3: (61.4, 2.4, 59.8, 2.4),
        6: (67.6, 2.5, 65.7, 2.5),
        9: (72.3, 2.7, 70.4, 2.7),
        12: (76.0, 2.8, 74.3, 2.8),
        18: (82.4, 3.1, 80.7, 3.1),
        24: (87.8, 3.4, 86.4, 3.4),
        36: (96.1, 3.8, 95.1, 3.9),
        48: (102.9, 4.2, 101.9, 4.3),
        60: (109.1, 4.5, 108.4, 4.6)
    }
    
    closest_age = min(references.keys(), key=lambda x: abs(x - age_months))
    
    if closest_age != age_months and abs(closest_age - age_months) <= 12:
        ages = sorted(references.keys())
        idx = ages.index(closest_age)
        if closest_age < age_months and idx < len(ages) - 1:
            next_age = ages[idx + 1]
        elif closest_age > age_months and idx > 0:
            next_age = ages[idx - 1]
            closest_age, next_age = next_age, closest_age
        else:
            if sex == 'M':
                return references[closest_age][0], references[closest_age][1]
            else:
                return references[closest_age][2], references[closest_age][3]
        
        age_diff = next_age - closest_age
        weight = (age_months - closest_age) / age_diff
        if sex == 'M':
            m1, sd1 = references[closest_age][0], references[closest_age][1]
            m2, sd2 = references[next_age][0], references[next_age][1]
        else:
            m1, sd1 = references[closest_age][2], references[closest_age][3]
            m2, sd2 = references[next_age][2], references[next_age][3]
        median = m1 + weight * (m2 - m1)
        sd = sd1 + weight * (sd2 - sd1)
        return median, sd
    
    if sex == 'M':
        return references[closest_age][0], references[closest_age][1]
    else:
        return references[closest_age][2], references[closest_age][3]

def calculate_height_for_age_z(row):
    """Calculate height-for-age z-score using WHO standards"""
    age = row['Age']
    height = row['Body Length']
    sex = row['Sex']
    median, sd = get_height_for_age_reference(age, sex)
    z_score = (height - median) / sd
    return z_score

@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    try:
        model = tf.keras.models.load_model('stunting_prediction_model.h5')
        preprocessor = joblib.load('stunting_preprocessor.joblib')
        return model, preprocessor, True
    except FileNotFoundError as e:
        st.error(f"Model atau preprocessor tidak ditemukan: {e}")
        st.error("Pastikan Anda sudah menjalankan notebook untuk melatih model terlebih dahulu!")
        return None, None, False

def predict_stunting(data, model, preprocessor):
    """Makes stunting predictions for new data using the trained model"""
    # Make a copy to avoid modifying the original
    data_copy = data.copy()
    
    # Apply feature engineering
    data_copy['BMI'] = data_copy['Body Weight'] / ((data_copy['Body Length']/100) ** 2)
    data_copy['Height_for_Age_Z'] = data_copy.apply(calculate_height_for_age_z, axis=1)
    
    # Get the z-score
    height_for_age_z = data_copy['Height_for_Age_Z'].values[0]
    
    # Determine WHO classification based on z-scores
    if height_for_age_z < -3:
        who_classification = 'Severely stunted (WHO)'
        who_stunting = 'Yes'
    elif height_for_age_z < -2:
        who_classification = 'Stunted (WHO)'
        who_stunting = 'Yes'
    else:
        who_classification = 'Not stunted (WHO)'
        who_stunting = 'No'
    
    # Preprocess data
    processed_data = preprocessor.transform(data_copy)
    
    # Make prediction
    stunting_probability = model.predict(processed_data).ravel()[0]
    
    return {
        'stunting_probability': float(stunting_probability),
        'stunting_prediction': who_stunting,
        'who_classification': who_classification,
        'height_for_age_z_score': float(height_for_age_z),
        'bmi': float(data_copy['BMI'].values[0])
    }

def create_z_score_chart(z_score):
    """Create a visual representation of z-score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = z_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Height-for-Age Z-Score"},
        delta = {'reference': -2},
        gauge = {
            'axis': {'range': [-4, 2]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-4, -3], 'color': "red"},
                {'range': [-3, -2], 'color': "orange"},
                {'range': [-2, 2], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': -2
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Aplikasi Prediksi Stunting Berdasarkan WHO Standards</h1>', unsafe_allow_html=True)
    
    # Load model and preprocessor
    model, preprocessor, model_loaded = load_model_and_preprocessor()
    
    if not model_loaded:
        st.stop()
    
    # Sidebar for input
    st.sidebar.markdown('<h2 class="sub-header">📊 Input Data Anak</h2>', unsafe_allow_html=True)
    
    # Input fields
    with st.sidebar.form("prediction_form"):
        st.markdown("### Data Demografi")
        sex = st.selectbox("Jenis Kelamin", ["M", "F"], format_func=lambda x: "Laki-laki" if x == "M" else "Perempuan")
        age = st.slider("Usia (bulan)", 0, 60, 24)
        asi_eksklusif = st.selectbox("ASI Eksklusif", ["Yes", "No"], format_func=lambda x: "Ya" if x == "Yes" else "Tidak")
        
        st.markdown("### Data Kelahiran")
        birth_weight = st.number_input("Berat Lahir (kg)", 1.0, 5.0, 3.0, step=0.1)
        birth_length = st.number_input("Panjang Lahir (cm)", 30.0, 60.0, 50.0, step=0.5)
        
        st.markdown("### Data Saat Ini")
        body_weight = st.number_input("Berat Badan Saat Ini (kg)", 2.0, 30.0, 12.0, step=0.1)
        body_length = st.number_input("Panjang/Tinggi Badan Saat Ini (cm)", 40.0, 120.0, 85.0, step=0.5)
        
        predict_button = st.form_submit_button("🔍 Prediksi Stunting", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            # Create input data
            input_data = pd.DataFrame({
                'Sex': [sex],
                'Age': [age],
                'Birth Weight': [birth_weight],
                'Birth Length': [birth_length],
                'Body Weight': [body_weight],
                'Body Length': [body_length],
                'ASI Eksklusif': [asi_eksklusif]
            })
            
            # Make prediction
            with st.spinner("Melakukan prediksi..."):
                result = predict_stunting(input_data, model, preprocessor)
            
            # Display results
            st.markdown('<h2 class="sub-header">📋 Hasil Prediksi</h2>', unsafe_allow_html=True)
            
            # Determine CSS class based on classification
            if "Severely stunted" in result['who_classification']:
                css_class = "severely-stunted"
                status_icon = "🚨"
            elif "Stunted" in result['who_classification']:
                css_class = "stunted"
                status_icon = "⚠️"
            else:
                css_class = "not-stunted"
                status_icon = "✅"
            
            # Main prediction box
            st.markdown(f"""
            <div class="prediction-box {css_class}">
                <h3>{status_icon} Status: {result['who_classification']}</h3>
                <p><strong>Probabilitas Model:</strong> {result['stunting_probability']:.1%}</p>
                <p><strong>Z-Score (Height-for-Age):</strong> {result['height_for_age_z_score']:.2f}</p>
                <p><strong>BMI:</strong> {result['bmi']:.2f} kg/m²</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional information
            st.markdown("### 📊 Interpretasi Hasil")
            
            col_interpret1, col_interpret2 = st.columns(2)
            
            with col_interpret1:
                st.markdown("**WHO Standards:**")
                st.markdown("- Z-Score ≥ -2: Normal")
                st.markdown("- Z-Score < -2: Stunted")
                st.markdown("- Z-Score < -3: Severely Stunted")
            
            with col_interpret2:
                if result['height_for_age_z_score'] < -3:
                    st.error("⚠️ Anak mengalami stunting berat. Konsultasi segera dengan tenaga kesehatan!")
                elif result['height_for_age_z_score'] < -2:
                    st.warning("⚠️ Anak mengalami stunting. Perlu perhatian khusus untuk gizi dan kesehatan.")
                else:
                    st.success("✅ Tinggi badan anak dalam kategori normal sesuai WHO.")
            
            # Z-Score visualization
            st.markdown("### 📈 Visualisasi Z-Score")
            z_score_chart = create_z_score_chart(result['height_for_age_z_score'])
            st.plotly_chart(z_score_chart, use_container_width=True)
    
    with col2:
        # Information panel
        st.markdown('<h3 class="sub-header">ℹ️ Informasi</h3>', unsafe_allow_html=True)
        
        st.info("""
        **Tentang Aplikasi Ini:**
        
        Aplikasi ini menggunakan model machine learning yang dilatih berdasarkan WHO Child Growth Standards untuk memprediksi risiko stunting pada anak.
        
        **Fitur Utama:**
        - Prediksi berdasarkan WHO standards
        - Kalkulasi Z-score Height-for-Age
        - Visualisasi hasil
        - Interpretasi mudah dipahami
        """)
        
        st.warning("""
        **⚠️ Penting:**
        
        Hasil prediksi ini hanya sebagai alat bantu dan tidak menggantikan diagnosis medis profesional. Selalu konsultasikan dengan tenaga kesehatan untuk evaluasi yang komprehensif.
        """)
        
        # Quick examples
        st.markdown("### 🔍 Contoh Cepat")
        
        if st.button("👶 Contoh Normal (12 bulan)", use_container_width=True):
            st.session_state.example_data = {
                'sex': 'M', 'age': 12, 'birth_weight': 3.2, 'birth_length': 50.0,
                'body_weight': 9.5, 'body_length': 75.0, 'asi_eksklusif': 'Yes'
            }
        
        if st.button("⚠️ Contoh Stunting (24 bulan)", use_container_width=True):
            st.session_state.example_data = {
                'sex': 'F', 'age': 24, 'birth_weight': 2.3, 'birth_length': 47.0,
                'body_weight': 8.5, 'body_length': 79.0, 'asi_eksklusif': 'No'
            }
    
    # Batch prediction section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📊 Prediksi Batch (Multiple Data)</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload file CSV dengan data anak", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Preview data yang diupload:")
            st.dataframe(batch_data.head())
            
            required_columns = ['Sex', 'Age', 'Birth Weight', 'Birth Length', 'Body Weight', 'Body Length', 'ASI Eksklusif']
            
            if all(col in batch_data.columns for col in required_columns):
                if st.button("🚀 Jalankan Prediksi Batch"):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, row in batch_data.iterrows():
                        single_data = pd.DataFrame([row])
                        result = predict_stunting(single_data, model, preprocessor)
                        results.append(result)
                        progress_bar.progress((i + 1) / len(batch_data))
                    
                    # Combine results
                    results_df = pd.DataFrame(results)
                    batch_data_with_results = pd.concat([batch_data, results_df], axis=1)
                    
                    st.success(f"✅ Berhasil memproses {len(batch_data)} data!")
                    st.dataframe(batch_data_with_results)
                    
                    # Download results
                    csv = batch_data_with_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Hasil Prediksi",
                        data=csv,
                        file_name="hasil_prediksi_stunting.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.markdown("### 📈 Ringkasan Hasil")
                    col_summary1, col_summary2, col_summary3 = st.columns(3)
                    
                    with col_summary1:
                        normal_count = sum(1 for r in results if "Not stunted" in r['who_classification'])
                        st.metric("Normal", normal_count, f"{normal_count/len(results)*100:.1f}%")
                    
                    with col_summary2:
                        stunted_count = sum(1 for r in results if "Stunted" in r['who_classification'] and "Severely" not in r['who_classification'])
                        st.metric("Stunted", stunted_count, f"{stunted_count/len(results)*100:.1f}%")
                    
                    with col_summary3:
                        severe_count = sum(1 for r in results if "Severely stunted" in r['who_classification'])
                        st.metric("Severely Stunted", severe_count, f"{severe_count/len(results)*100:.1f}%")
            else:
                st.error(f"File CSV harus memiliki kolom: {', '.join(required_columns)}")
        
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")

if __name__ == "__main__":
    main()
