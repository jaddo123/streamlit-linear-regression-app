import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Sleep and Study Habits Explorer", layout="wide")

# Simple custom dataset
@st.cache_data
def load_data():
    data = {
        "sleep_hours": [5, 6, 7, 8, 4, 9, 6, 7, 5, 8],
        "study_hours": [2, 3, 4, 5, 1, 6, 3, 4, 2, 5],
        "coffee_cups": [3, 2, 2, 1, 4, 1, 3, 2, 4, 1],
        "attendance": [70, 80, 85, 90, 60, 95, 75, 88, 65, 92],
        "exam_score": [65, 70, 78, 85, 55, 92, 72, 80, 60, 88]
    }
    return pd.DataFrame(data)


df = load_data()
features = [col for col in df.columns if col != "exam_score"]

st.title("Sleep and Study Habits Explorer")
st.write("Explore how different student habits relate to exam scores.")

# Interactive feature 1
x_feature = st.selectbox("Choose an x-variable:", features)

# Interactive feature 2
point_size = st.slider("Choose point size:", 20, 100, 50)

# Interactive feature 3
show_data = st.checkbox("Show dataset")

if show_data:
    st.dataframe(df)

X = df[[x_feature]]
y = df["exam_score"]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

fig, ax = plt.subplots()
ax.scatter(df[x_feature], y, s=point_size, alpha=0.7)
ax.plot(df[x_feature], y_pred, linewidth=2)
ax.set_xlabel(x_feature)
ax.set_ylabel("exam_score")
ax.set_title(f"{x_feature} vs exam_score")

st.pyplot(fig)

st.write(
    f"Regression equation: exam_score = "
    f"{model.coef_[0]:.2f} * {x_feature} + {model.intercept_:.2f}"
)