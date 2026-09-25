# 1. Unstage the corrupted references and add the fresh script version
git add streamlit_app.py

# 2. Record the fixed tracking commit marker message
git commit -m "fix: replaced notebook json structure with native raw python text"

# 3. Securely upload the files online
git push origin main
