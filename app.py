# app.py

import streamlit as st
import pandas as pd

# Set the title of the web app
st.title("🏏 IPL Matches Dashboard")

# Load the IPL dataset
try:
    df = pd.read_csv("IPL_Matches.csv")
    st.success("✅ IPL dataset loaded successfully!")
except FileNotFoundError:
    st.error("❌ IPL_Matches.csv not found. Please make sure the file is in the same folder as app.py.")
    st.stop()

# Show available columns (for debugging)
st.write("📌 Columns in the dataset:", df.columns.tolist())

# Check if 'Season' column exists
if 'Season' in df.columns:
    # Dropdown to select a season
    seasons = df['Season'].dropna().unique().tolist()
    selected_season = st.selectbox("📅 Select a Season", sorted(seasons))

    # Filter data by selected season
    filtered_df = df[df['Season'] == selected_season]

    # Display the filtered matches
    st.subheader(f"📄 Match Data for Season {selected_season}")
    st.write(filtered_df)

    # Total number of matches in selected season
    st.info(f"Total Matches Played in {selected_season}: {filtered_df.shape[0]}")
else:
    st.error("❌ 'Season' column not found in the dataset.")
