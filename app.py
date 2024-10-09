import streamlit as st
import Employee  # Import functions from em.py

st.title("Predictive Attrition Analytics")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Employee Attrition CSV", type=["csv"])
if uploaded_file:
    # Load data from uploaded file
    data = Employee.load_data(uploaded_file)

    # Display data information
    Employee.display_data_info(data)

    # Plot charts for data visualization
    st.subheader("Attrition Visualizations")
    Employee.plot_charts(data)

    # Process and encode the data
    data = Employee.process_and_encode_data(data)

    # Train and evaluate the model
    st.subheader("Model Training and Evaluation")
    accuracy = Employee.train_and_evaluate_model(data)
    st.write(f"Model Accuracy: {accuracy}")
else:
    st.write("Please upload a CSV file to proceed.")
