import base64
import streamlit as st
import pickle

# @st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('image.jpg')

# Load the trained model
with open('model3.pkl', 'rb') as f:
    model = pickle.load(f)

# Define diagnosis mapping dictionary
diagnoses = {
    0: 'Negative',
    1: 'Positive',
}

# Predicted diagnosis color
diagnosis_color = 'white'
title_color = 'white'  # Title color
title_css = f"<h1 style='text-align: center; color: {title_color};'>Thyroid Diagnosis Predictor</h1>"

# Detect button color
detect_button_color = 'white'

# Function to preprocess inputs before prediction
def preprocess_inputs(age, sex, sick, pregnant,
                      thyroid_surgery, I131_treatment, lithium,
                      goitre, tumor, TSH, T3, TT4, T4U, FTI):

    # Replace 'Yes' with 1 and 'No' with 0
    binary_map = {'Yes': 1, 'No': 0, '': None}
    sick = binary_map.get(sick)
    pregnant = binary_map.get(pregnant)
    thyroid_surgery = binary_map.get(thyroid_surgery)
    I131_treatment = binary_map.get(I131_treatment)
    lithium = binary_map.get(lithium)
    goitre = binary_map.get(goitre)
    tumor = binary_map.get(tumor)

    # Replace 'M' and 'F' with binary 0 and 1
    sex = 1 if sex == 'F' else 0 if sex == 'M' else None


    return [age, sex, sick, pregnant,
            thyroid_surgery, I131_treatment, lithium,
            goitre, tumor, TSH, T3, TT4, T4U, FTI]


# Function to predict the diagnosis based on inputs
def predict_diagnosis(inputs):
    # Assuming 'model' is a trained machine learning model
    # Replace 'model.predict()' with the actual function to make predictions
    output = model.predict([inputs])[0]
    return output


# Streamlit app
def main():
    # Title
    st.markdown(title_css, unsafe_allow_html=True)
    
    # st.markdown(
    #     """
    #     <style>
    #         [data-testid="stAppViewContainer"] > .main {
    #             background-image: url("data:image/png;base64,{img}");
    #             background-size: cover;
    #             background-repeat: no-repeat;
    #             background-attachment: fixed;
    #             background-position: center;
    #         }

    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

    # Sidebar
    st.sidebar.title("About Project :")
    st.sidebar.write("This Streamlit app serves as a Thyroid Diagnosis Predictor. It utilizes machine learning to predict thyroid diagnosis based on various patient attributes such as age, sex, medical history, and laboratory test results. Users can input patient data and receive an immediate diagnosis prediction, helping medical professionals make informed decisions efficiently.")

    st.sidebar.title("Attributes Information :")
    st.sidebar.write("""
        - Age: Age of the patient (int)
        - Sex: Gender of the patient (str)
        - Sick: Whether patient is sick (bool)
        - Pregnant: Whether patient is pregnant (bool)
        - Thyroid Surgery: Whether patient has undergone thyroid surgery (bool)
        - I131 Treatment: Whether patient is undergoing I131 treatment (bool)
        - Lithium: Whether patient takes lithium (bool)
        - Goitre: Whether patient has goitre (bool)
        - Tumor: Whether patient has tumor (bool)
        - TSH: TSH level in blood from lab work (float)
        - T3: T3 level in blood from lab work (float)
        - TT4: TT4 level in blood from lab work (float)
        - T4U: T4U level in blood from lab work (float)
        - FTI: FTI level in blood from lab work (float)
    """)

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', value=None)
        pregnant = st.selectbox('Pregnant', options=['No', 'Yes'])
        lithium = st.selectbox('Lithium', options=['No', 'Yes'])
        TSH = st.number_input('TSH', value=None)
        T4U = st.number_input('T4U', value=None)


    with col2:
        sex = st.selectbox('Sex', options=['M', 'F'])
        thyroid_surgery = st.selectbox('Thyroid Surgery', options=['No', 'Yes'])
        goitre = st.selectbox('Goitre', options=['No', 'Yes'])
        T3 = st.number_input('T3', value=None)
        FTI = st.number_input('FTI', value=None)


    with col3:
        sick = st.selectbox('Sick', options=['No', 'Yes'])
        I131_treatment = st.selectbox('I131 Treatment', options=['No', 'Yes'])
        tumor = st.selectbox('Tumor', options=['No', 'Yes'])
        TT4 = st.number_input('TT4', value=None)

    # Detect button
    with col2:
        detect_button = st.button('Detect', key='predict_button')
        detect_button_container = st.container()
        with detect_button_container:
            detect_button_css = f"""
                <style>
                    .stButton > button:first-child {{
                        width: 100%;
                        color: white;
                        border-color: white;
                        border-radius: 5px;
                        padding: 10px;
                        margin-top: 30px;
                    }}
                </style>
            """
            st.markdown(detect_button_css, unsafe_allow_html=True)

        if detect_button:
            # Preprocess inputs
            inputs = preprocess_inputs(age, sex, sick,
                                       pregnant,
                                       thyroid_surgery, I131_treatment,
                                       lithium,
                                       goitre, tumor, TSH, T3, TT4, T4U, FTI)
            # Get prediction
            diagnosis_num = predict_diagnosis(inputs)
            diagnosis_label = diagnoses.get(diagnosis_num, 'Unknown')
            st.markdown(
                f"<h1 style='text-align: center; color: {diagnosis_color};'>{diagnosis_label}</h1>",
                unsafe_allow_html=True)


if __name__ == '__main__':
    main()
