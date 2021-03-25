
import pandas as pd 
import numpy as np 
import streamlit as st 
import base64 
import seaborn as sns
import matplotlib.pyplot as plt

st.title("NBA Player Stats Explorer")

st.sidebar.header("User Input Features")
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950, 2020))))

@st.cache
def load_df(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == "Age"].index)
    raw = raw.fillna(0)
    playerstats = raw.drop(["Rk"], axix=1)
    return playerstats
playerstats = load_df(selected_year)

sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect("Team", sorted_unique_team, sorted_unique_team)

unique_pos = ["C", "PF", "SF", "PG", "SG"]
selected_pos = st.sidebar.multiselect("Position", unique_pos, unique_pos)

df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Tm.isin(selected_pos))]

st.header("Display Player Stats of Selectes Team(s)")
st.write("Data Dimension: " +str(df_selected_team.shape[0]) + " rows and " + str(df_selected_team.shape[1] + " columns"))
st.dataframe(df_selected_team)

def files(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(files(df_selected_team), unsafe_allow_html=True)

#heatmap

if st.button("Intercorrelation Heatmap"):
    st.header("Intercorrelation Matrix Heatmap")
    df_selected_team.to_csv("output.csv", index=False)
    df = pd.read_csv("output.csv")

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_strle("white"):
        f, ax = plt.subplot(figsize=(7, 7))
        ax = sns.heatmap(corr, mask=mask, vamx=1, square=True)
    st.pyplot()


