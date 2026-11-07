import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="🛒",
    layout="wide"
)
# 2. Load Dataset and Model
df = pd.read_csv(
    "dataset/retail_sales_forecast_dataset.csv"
)
model = joblib.load(
    "model.joblib"
)
# 3. Main Title

st.title("🛒 Retail Sales Forecasting")
st.write(
    "Predict retail sales using Machine Learning."
)

st.divider()

# 4. Sales Overview
st.subheader("📊 Sales Overview")
col1, col2, col3 = st.columns(3)
col1.metric(
    "Total Units Sold",
    f"{int(df['Units_Sold'].sum()):,}"
)
col2.metric(
    "Average Sales",
    f"{df['Units_Sold'].mean():.2f}"
)
col3.metric(
    "Total Records",
    f"{len(df):,}"
)
st.divider()

# 5. Sidebar Input
st.sidebar.title("Sales Information")
Year = st.sidebar.selectbox(
    "Enter Year",
    sorted(
        df["Year"].dropna().unique()
    )
)
Month = st.sidebar.selectbox(

    "Enter Month",

    list(range(1, 13))

)
Product_Category = st.sidebar.selectbox(
    "Product Category",
    sorted(
        df["Product_Category"].dropna().unique()
    )
)
Region = st.sidebar.selectbox(
    "Region",
    sorted(
        df["Region"].dropna().unique()
    )
)
Weather_Condition = st.sidebar.selectbox(
    "Weather Condition",
    sorted(
        df["Weather_Condition"].dropna().unique()
    )
)
Season = st.sidebar.selectbox(
    "Season",
    sorted(
        df["Season"].dropna().unique()
    )
)
Inventory_Level = st.sidebar.number_input(
    "Inventory Level",
    min_value=0,
    value=100
)
Unit_Price = st.sidebar.number_input(
    "Unit Price",
    min_value=0.0,
    value=50.0
)
Discount_Percent = st.sidebar.number_input(
    "Discount Percentage",
    min_value=0.0,
    max_value=100.0,
    value=0.0
)
Promotion = st.sidebar.selectbox(
    "Promotion",
    sorted(
        df["Promotion"].dropna().unique()
    )
)
Competitor_Price = st.sidebar.number_input(
    "Competitor Price",
    min_value=0.0,
    value=50.0
)

# 6. Create Input Data

input_values = pd.DataFrame([{
    "Year": Year,
    "Month": Month,
    "Product_Category": Product_Category,
    "Region": Region,
    "Weather_Condition": Weather_Condition,
    "Season": Season,
    "Inventory_Level": Inventory_Level,
    "Unit_Price": Unit_Price,
    "Discount_Percent": Discount_Percent,
    "Promotion": Promotion,
    "Competitor_Price": Competitor_Price
}])

# 7. Display Input Information
st.subheader("📝 Sales Information")
st.dataframe(
    input_values,
    width="stretch"
)
st.divider()

# 8. Sales Prediction

st.subheader("🔮 Sales Prediction")
if st.button(
    "Predict Sales",
    type="primary"
):
    prediction = model.predict(
        input_values
    )[0]
    prediction = max(
        0,
        round(float(prediction))
    )
    # Estimated Revenue
    estimated_revenue = (
        prediction * Unit_Price
    )
    # Display Result
    st.success(
        "Sales Prediction Completed!"
    )
    col1, col2 = st.columns(2)
    col1.metric(

        "Predicted Units Sold",

        f"{prediction} Products"
    )
    col2.metric(
        "Estimated Revenue",
        f"₹{estimated_revenue:,.2f}"
    )
    # Prediction Details
    result = pd.DataFrame([{
        "Year": Year,
        "Month": Month,
        "Product Category": Product_Category,
        "Region": Region,
        "Predicted Units Sold": prediction,
        "Unit Price": Unit_Price,
        "Estimated Revenue": estimated_revenue
    }])
    st.subheader("📋 Prediction Details")
    st.dataframe(
        result,
        width="stretch"
    )
    # Download Result
    csv = result.to_csv(
        index=False
    ).encode("utf-8")
    st.download_button(
        label="Download Prediction",
        data=csv,
        file_name="sales_prediction.csv",
        mime="text/csv"
    )
st.divider()
# 9. Historical Sales Graph
st.subheader("📈 Historical Sales Trends")
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(

        df["Date"],

        errors="coerce"
    )
    sales = df.dropna(
        subset=["Date"]
    ).copy()
    monthly_sales = (
        sales.set_index("Date")
        ["Units_Sold"]
        .resample("ME")
        .sum()
    )
    st.line_chart(
        monthly_sales
    )
else:
    st.info(
        "Date column is not available."
    )
# 10. Footer
st.divider()
st.caption(
    "Retail Sales Forecasting | "
    "Machine Learning | "
    "Random Forest Regression"
)