import streamlit as st
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy as dc
st.title('Estimate BHP')
st.subheader("Upload your CSV file here")
required_columns = ['PRODUCTION DATE', 'Qliquid', 'GOR', 'Pwh', 'THT', 'WCT']
uploaded_file = st.file_uploader("Choose a file")
def prepare_dataframe_for_lstm(df, n_steps):
    date = df['PRODUCTION DATE']
    date = date[2]
    df = dc(df)
    df.set_index('PRODUCTION DATE', inplace=True)

    for i in range(1, n_steps + 1):
        df[f'Qliquid(t-{i})'] = df['Qliquid'].shift(i)
        df[f'GOR(t-{i})'] = df['GOR'].shift(i)
        df[f'Pwh(t-{i})'] = df['Pwh'].shift(i)
        df[f'THT(t-{i})'] = df['THT'].shift(i)
        df[f'WCT(t-{i})'] = df['WCT'].shift(i)

    df.dropna(inplace=True)
    return df

# File processing and validation
if uploaded_file is not None:
    try:
        dataframe = pd.read_csv(uploaded_file)
        missing_columns = [col for col in required_columns if col not in dataframe.columns]

        if missing_columns:
            st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_columns)}")
            st.image("description.jpg", caption="Please check that the uploaded file has this structure")
        else:
            dataframe = prepare_dataframe_for_lstm(dataframe, 2)
            st.success("File successfully uploaded and verified!")
            st.write(dataframe)
            st.session_state['dataframe'] = dataframe  # Save the dataframe in session state
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

# Load the saved model and scaler
def load_model():
    with open('modelXAMATR_MODEL.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
model = data['model']
scaler = data['scaler']



def manual_unscale(scaled_data):
    # Extract the mean and standard deviation from the scaler
    means = 3255
    stds = 253

    # Ensure scaled_data is in the correct shape for unscaling
    scaled_data = np.array(scaled_data)

    # Rescale the data back to its original scale
    unscaled_data = (scaled_data * stds) + means

    return unscaled_data
def start_prediction():
    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']
        
        # Apply scaling to the necessary columns
        columns_to_scale = ['MBHFP','Qliquid', 'GOR', 'Pwh', 'THT','WCT','Qliquid(t-1)','GOR(t-1)','Pwh(t-1)','THT(t-1)','WCT(t-1)','Qliquid(t-2)','GOR(t-2)','Pwh(t-2)','THT(t-2)',	'WCT(t-2)']

        # Ensure these columns exist in the DataFrame
        scaled_columns = [col for col in columns_to_scale if col in df.columns]
        data_predicted = scaler.transform(df[scaled_columns])

        # Convert back to a DataFrame to preserve column labels (optional)
        scaled_df = pd.DataFrame(data_predicted, columns=scaled_columns)

        # Prepare the input for the LSTM
        X= scaled_df[['Qliquid', 'GOR', 'Pwh', 'THT','WCT','Qliquid(t-1)','GOR(t-1)','Pwh(t-1)','THT(t-1)','WCT(t-1)','Qliquid(t-2)','GOR(t-2)','Pwh(t-2)','THT(t-2)',	'WCT(t-2)']]
        X = X.values.reshape((scaled_df.shape[0], 5, 3))  # Adjust dimensions as necessary
        y_pred = model.predict(X)
        y_pred = y_pred.reshape(1,-1)
        y_pred_1 = y_pred.reshape(-1)
        unscaled_data = manual_unscale(y_pred_1)
        st.session_state['prediction_result'] = unscaled_data
    else:
        st.error("Please upload a file and preprocess it first.")


# Button to start prediction
if st.button("Predict"):
    start_prediction()


# Display the prediction result
if 'prediction_result' in st.session_state:
    st.subheader("Prediction Result:")
    st.write(st.session_state['prediction_result'])

    # Create a pandas DataFrame for the prediction results to use with st.linechart
    prediction_df = pd.DataFrame(st.session_state['prediction_result'], columns=['Predicted BHP'])


