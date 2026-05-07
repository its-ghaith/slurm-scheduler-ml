from pathlib import Path

import streamlit as st

from countcv.focidet.gui.foci_counter import init_logo

init_logo()

wiki_path = Path("src") / "countcv" / "focidet" / "gui" / "wiki.md"
with open(wiki_path) as f:
	wiki_page = f.read()

st.markdown(wiki_page)
