ls -la
# 1. Force stage both the script and the model binary file
git add -f streamlit_app.py models/artifacts.pkl requirements.txt .gitignore

# 2. Apply a clean tracking commit message
git commit -m "deploy: root directory streamlit configuration sync with model binary weights"

# 3. Securely push to the cloud master main branch
git push origin main
