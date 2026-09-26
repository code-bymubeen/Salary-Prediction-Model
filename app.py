import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    page_icon="💸",
    layout="wide"
)


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Salary_Data.csv")


# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("salary_prediction_model.pkl")


df = load_data()
model = load_model()


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("💰 Salary Prediction")

page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Salary Analysis", "Salary Prediction"]
)


# Remove rows where Salary is missing
salary_data = df.dropna(subset=["Salary"])


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("💰 Salary Prediction Dashboard")
    st.write(
        "Explore salary trends and understand how experience, "
        "education and job roles relate to salary."
    )

    # -----------------------------
    # KPI CARDS
    # -----------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Job Titles",
        df["Job Title"].nunique()
    )

    col3.metric(
        "Average Salary",
        f"${salary_data['Salary'].mean():,.0f}"
    )

    col4.metric(
        "Average Experience",
        f"{df['Years of Experience'].mean():.1f} years"
    )

    st.divider()

    # -----------------------------
    # SALARY DISTRIBUTION
    # -----------------------------
    st.subheader("Salary Distribution")

    fig_salary = px.histogram(
        salary_data,
        x="Salary",
        nbins=40,
        title="Distribution of Salaries"
    )

    st.plotly_chart(
        fig_salary,
        use_container_width=True
    )

    # -----------------------------
    # EXPERIENCE VS SALARY
    # -----------------------------
    st.subheader("Experience vs Salary")

    fig_exp = px.scatter(
        salary_data,
        x="Years of Experience",
        y="Salary",
        color="Education Level",
        hover_data=[
            "Age",
            "Job Title",
            "Gender"
        ],
        title="Experience vs Salary"
    )

    st.plotly_chart(
        fig_exp,
        use_container_width=True
    )


# =========================================================
# SALARY ANALYSIS
# =========================================================

elif page == "Salary Analysis":

    st.title("📊 Salary Analysis")

    # -----------------------------
    # EDUCATION VS SALARY
    # -----------------------------
    st.subheader("Average Salary by Education Level")

    education_salary = (
        salary_data
        .groupby("Education Level")["Salary"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig_education = px.bar(
        education_salary,
        x="Education Level",
        y="Salary",
        title="Average Salary by Education Level"
    )

    st.plotly_chart(
        fig_education,
        use_container_width=True
    )

    # -----------------------------
    # TOP JOB TITLES
    # -----------------------------
    st.subheader("Top 15 Job Titles by Average Salary")

    job_salary = (
        salary_data
        .groupby("Job Title")
        .agg(
            Average_Salary=("Salary", "mean"),
            Number_of_People=("Salary", "count")
        )
        .sort_values(
            "Average_Salary",
            ascending=False
        )
        .head(15)
        .reset_index()
    )

    fig_jobs = px.bar(
        job_salary,
        x="Average_Salary",
        y="Job Title",
        orientation="h",
        hover_data=["Number_of_People"],
        title="Top 15 Job Titles"
    )

    st.plotly_chart(
        fig_jobs,
        use_container_width=True
    )

    # -----------------------------
    # GENDER VS SALARY
    # -----------------------------
    st.subheader("Average Salary by Gender")

    gender_salary = (
        salary_data
        .groupby("Gender")["Salary"]
        .mean()
        .reset_index()
    )

    fig_gender = px.bar(
        gender_salary,
        x="Gender",
        y="Salary",
        title="Average Salary by Gender"
    )

    st.plotly_chart(
        fig_gender,
        use_container_width=True
    )


# =========================================================
# SALARY PREDICTION
# =========================================================

elif page == "Salary Prediction":

    st.title("🔮 Salary Prediction")

    st.write(
        "Enter the details below to estimate the expected salary."
    )

    # -----------------------------
    # GET OPTIONS FROM REAL DATA
    # -----------------------------
    job_titles = sorted(
        df["Job Title"]
        .dropna()
        .unique()
        .tolist()
    )

    education_levels = sorted(
        df["Education Level"]
        .dropna()
        .unique()
        .tolist()
    )

    genders = sorted(
        df["Gender"]
        .dropna()
        .unique()
        .tolist()
    )

    # -----------------------------
    # INPUTS
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        experience = st.number_input(
            "Years of Experience",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5
        )

        gender = st.selectbox(
            "Gender",
            genders
        )

    with col2:

        education = st.selectbox(
            "Education Level",
            education_levels
        )

        job_title = st.selectbox(
            "Job Title",
            job_titles
        )

    st.divider()

    # -----------------------------
    # PREDICTION BUTTON
    # -----------------------------
    if st.button(
        "Predict Salary",
        type="primary"
    ):

        # Create input DataFrame
        input_data = pd.DataFrame({
            "Age": [age],
            "Years of Experience": [experience],
            "Job Title": [job_title],
            "Education Level": [education],
            "Gender": [gender]
        })

        # Predict
        prediction = model.predict(input_data)

        predicted_salary = prediction[0]

        # -----------------------------
        # SHOW RESULT
        # -----------------------------
        st.success("Prediction completed!")

        st.metric(
            "Estimated Salary",
            f"${predicted_salary:,.0f}"
        )

        st.write("### Input Details")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.caption(
    "Salary Prediction Dashboard | Built with Python, "
    "Scikit-learn, Pandas, Plotly and Streamlit,| "
    "Built by Mubeen Shehzadi"
)
