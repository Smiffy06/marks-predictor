import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Load your CSV files
df1 = pd.read_csv('semester1.csv')
df2 = pd.read_csv('semester2.csv')
df3 = pd.read_csv('semester3.csv')

# Title
st.title("Result Prediction Dashboard")
st.write("---")
st.subheader("Select Semester and Chart Type")
st.write("---")
col1, col2 = st.columns(2)
with col1:
    semester_selection = st.radio("Select Semester:", options=["Semester 1", "Semester 2", "Semester 3", "All Semesters"])
with col2:
    chart_selection = st.radio("Select Chart Type:", options=["Bar Chart", "Line Chart", "Sunburst Chart"])

# Function to create a bar chart
def create_bar_chart(df, sem_title):
    fig = px.bar(df, x='Subject', y='Percentage', color='Subject', title=f'Result for {sem_title}',
                labels={'Subject': 'Subjects', 'Percentage': 'Percentage'})
    fig.update_xaxes(showticklabels=False)
    return fig

# Function to create a line chart
def create_line_chart(df, sem_title):
    fig = px.line(df, x='Subject', y='Percentage', title=f'Result for {sem_title}',)
    fig.update_xaxes(showticklabels=True)
    return fig

# Function to create a sunburst chart
def create_sunburst_chart(df, sem_title):
    fig = px.sunburst(
        df,title=f'Result for {sem_title}',
        path=["Subject","Percentage"],
        values="Percentage",
        color="Subject",
        hover_data=["Percentage"])
    return fig
st.write("---")
# Display based on semester and chart type
if semester_selection == "Semester 1":
    if chart_selection == 'Bar Chart':
        st.plotly_chart(create_bar_chart(df1, "Semester 1"))
    elif chart_selection == 'Line Chart':
        st.plotly_chart(create_line_chart(df1, "Semester 1"))
    elif chart_selection == 'Sunburst Chart':
        st.plotly_chart(create_sunburst_chart(df1, "Semester 1"))

elif semester_selection == "Semester 2":
    if chart_selection == 'Bar Chart':
        st.plotly_chart(create_bar_chart(df2, "Semester 2"))
    elif chart_selection == 'Line Chart':
        st.plotly_chart(create_line_chart(df2, "Semester 2"))
    elif chart_selection == 'Sunburst Chart':
        st.plotly_chart(create_sunburst_chart(df2, "Semestere 2"))

elif semester_selection == "Semester 3":
    if chart_selection == 'Bar Chart':
        st.plotly_chart(create_bar_chart(df3, "Semester 3"))
    elif chart_selection == 'Line Chart':
        st.plotly_chart(create_line_chart(df3, "Semester 3"))
    elif chart_selection == 'Sunburst Chart':
        st.plotly_chart(create_sunburst_chart(df3, "Semester 3"))

elif semester_selection == "All Semesters":
    col1, col2, col3 = st.columns(3)
    if chart_selection == 'Bar Chart':
        with col1:
            st.subheader("Semester 1")
            st.plotly_chart(create_bar_chart(df1, "Semester 1"))
        with col2:
            st.subheader("Semester 2")
            st.plotly_chart(create_bar_chart(df2, "Semester 2"))
        with col3:
            st.subheader("Semester 3")
            st.plotly_chart(create_bar_chart(df3, "Semester 3"))
    elif chart_selection == 'Line Chart':
        with col1:
            st.subheader("Semester 1")
            st.plotly_chart(create_line_chart(df1, "Semester 1"))
        with col2:
            st.subheader("Semester 2")
            st.plotly_chart(create_line_chart(df2, "Semester 2"))
        with col3:
            st.subheader("Semester 3")
            st.plotly_chart(create_line_chart(df3, "Semester 3"))
    elif chart_selection == 'Sunburst Chart':
        with col1:
            st.subheader("Semester 1")
            st.plotly_chart(create_sunburst_chart(df1, "Semester 1"))
        with col2:
            st.subheader("Semester 2")
            st.plotly_chart(create_sunburst_chart(df2, "Semester 2"))
        with col3:
            st.subheader("Semester 3")
            st.plotly_chart(create_sunburst_chart(df3, "Semester 3"))
st.write("---")
#predict the result for semester 4
st.subheader("Predicting Result for Semester 4")
p1=round(sum(df1['Marks '])/sum(df1['Total'])*100)
p2=round(sum(df2['Marks '])/sum(df2['Total'])*100)
p3=round(sum(df3['Marks '])/sum(df3['Total'])*100)

df4=pd.DataFrame({
    'Semester': ['1', '2', '3'],'Percentage': [p1, p2, p3]
})

X = df4[['Semester']]
y = df4['Percentage']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict semester 4
sem4 = model.predict([[4]])  # Predict for semester 4
predicted_sem4 = round(sem4[0], 2)

# Add to DataFrame
new_row = pd.DataFrame([{
    'Percentage': predicted_sem4,
    'Semester': 4
}])

df4 = pd.concat([df4, new_row], ignore_index=True)

st.success(f"Predicted Result for Semester 4: {predicted_sem4}%")

# Display the DataFrame with predicted result
st.dataframe(df4,width=800, height=177)
st.write("---")
st.subheader("Result Prediction Chart (All Semesters)")
col4,col5,col6 = st.columns(3)
with col4:
    #barchart
    st.plotly_chart(px.bar(df4, x="Semester", y="Percentage",color="Semester"))
with col5:
    #linechart
    st.plotly_chart(px.line(df4, x="Semester", y="Percentage"))
with col6:
    #sunburst chart
    st.plotly_chart(px.sunburst(df4, path=["Semester","Percentage"], values="Percentage"))
st.write("---")
#st.balloons()