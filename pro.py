import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(page_title="Sales Insights", layout="wide")

# Database connection
conn = sqlite3.connect("sales_feedback.db", check_same_thread=False)
c = conn.cursor()

# Create feedback table if not exists
c.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        phone TEXT,
        feedback TEXT,
        sentiment TEXT
    )
""")
conn.commit()

# Sample user database (replace with actual database in production)
users_db = {
    "Maha": {"password": "Streamlit@24"}
}


def login():
    """User login function."""
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users_db and users_db[username]['password'] == password:
            st.success("Login successful!")
            st.session_state["logged_in"] = True
            return True
        else:
            st.error("Invalid credentials. Please try again.")
            return False
    return False


def signup():
    """User signup function."""
    st.title("Sign Up Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        elif username in users_db:
            st.error("Username already exists. Please choose another one.")
        else:
            users_db[username] = {'password': password}
            st.success("Sign up successful! You can now log in.")


def analyze_sentiment(text):
    """Analyze text sentiment using TextBlob."""
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        return "Positive"
    elif blob.sentiment.polarity < 0:
        return "Negative"
    return "Neutral"


def save_feedback_to_db(name, phone, feedback, sentiment):
    """Save feedback to the SQLite database."""
    c.execute(
        "INSERT INTO feedback (name, phone, feedback, sentiment)"
        " VALUES (?, ?, ?, ?)",
        (name, phone, feedback, sentiment),
    )
    conn.commit()


def load_feedback_from_db():
    """Retrieve feedback data from the database."""
    c.execute("SELECT name, phone, feedback, sentiment FROM feedback")
    rows = c.fetchall()
    return pd.DataFrame(rows, columns=["Name", "Phone", "Feedback",
                                       "Sentiment"])


def customer_feedback_section():
    """Display customer feedback and sentiment analysis."""
    st.subheader("Customer Feedback")

    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        phone = st.text_input("Your Phone Number")
        feedback = st.text_area("Your Feedback")
        submit_feedback = st.form_submit_button("Submit Feedback")

        if submit_feedback and feedback:
            sentiment = analyze_sentiment(feedback)
            save_feedback_to_db(name, phone, feedback, sentiment)
            st.success("Thank you for your feedback!")

    feedback_data = load_feedback_from_db()

    if not feedback_data.empty:
        st.write("### Customer Feedback and Sentiment Analysis")
        st.dataframe(feedback_data)

        sentiment_counts = feedback_data["Sentiment"].value_counts()
        fig = px.bar(
            sentiment_counts,
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            title="Sentiment Distribution",
        )
        st.plotly_chart(fig)


def upload_and_analysis():
    """Upload and analyze sales data."""
    st.title("Sales and Performance Analytics")
    uploaded_file = st.file_uploader("Upload your CSV/Excel file",
                                     type=['csv', 'xlsx'])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Data Overview")
        st.write(df.head())

        chart_type = st.selectbox("Choose chart type", ["Bar Chart",
                                                        "Line Plot",
                                                        "Scatter Plot",
                                                        "Pie Chart"])

        if chart_type != "Pie Chart":
            x_axis = st.selectbox("Select X-axis", df.columns)
            y_axis = st.selectbox("Select Y-axis", df.columns)

        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        elif chart_type == "Line Plot":
            fig = px.line(df, x=x_axis, y=y_axis,
                          title=f"{y_axis} vs {x_axis}")
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis,
                             title=f"{y_axis} vs {x_axis}")
        elif chart_type == "Pie Chart":
            pie_column = st.selectbox(
                "Select Column for Pie Chart", df.columns)
            fig = px.pie(df, names=pie_column,
                         title="Pie Chart of {pie_column}")

        st.plotly_chart(fig)


def sales_prediction(df):
    """Perform sales prediction using linear regression."""
    st.write("### Sales Prediction Model")

    if "Sales" in df.columns and "Revenue" in df.columns:
        X, y = df[["Revenue"]], df["Sales"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_data = pd.DataFrame({"Actual Sales": y_test,
                                  "Predicted Sales": y_pred})
        st.write(test_data.head())

        fig = px.scatter(
            x=y_test, y=y_pred,
            title="Predicted vs Actual Sales",
            labels={'x': "Actual Sales", 'y': "Predicted Sales"}
        )
        fig.add_shape(
            type="line",
            x0=y_test.min(), x1=y_test.max(),
            y0=y_test.min(), y1=y_test.max(),
            line=dict(color="Red", dash="dash")
        )
        st.plotly_chart(fig)
    else:
        st.warning(
            "Dataset must contain 'Sales' and 'Revenue' "
            "columns for prediction.")


if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        if login():
            st.rerun()

    else:
        page = st.sidebar.radio("Navigation", ["Customer Feedback",
                                               "Upload & Analysis",
                                               "Sales Prediction"])

        if page == "Customer Feedback":
            customer_feedback_section()
        elif page == "Upload & Analysis":
            upload_and_analysis()
        elif page == "Sales Prediction":
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                sales_prediction(df)
