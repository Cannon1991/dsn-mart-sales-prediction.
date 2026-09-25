# 1. Unstage the corrupted references and add the fresh script version
git add streamlit_app.py

# 2. Record the fixed tracking commit marker message
git commit -m "fix: replaced notebook json structure with native raw python text"

# 3. Securely upload the files online
git push origin main
git add README.md
git commit -m "docs: finalized public readme portfolio documentation matrix"
git push origin main
# 1. Pull potential online modifications down safely
!git pull origin main --rebase

# 2. Tell git to stop tracking old or extra script versions
!git rm -f app.py model_pipeline.py 2>/dev/null

# 3. Add your fresh dashboard file configuration
!git add streamlit_app.py

# 4. Save and commit your changes
!git commit -m "fix: explicit deployment codebase structural realignment patch"

# 5. Push up the clean repo state
!git push origin main --force
