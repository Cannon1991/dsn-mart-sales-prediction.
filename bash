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
# 1. Fetch online updates to stay completely in sync
!git pull origin main --rebase

# 2. Force remove old file configurations from git tracking memory
!git rm -f app.py model_pipeline.py 2>/dev/null

# 3. Add the two newly verified files
!git add streamlit_app.py requirements.txt

# 4. Commit and upload directly to GitHub
!git commit -m "fix: production app release sync"
!git push origin main --force
# 1. Fetch remote changes to stay aligned with the cloud branch
!git pull origin main --rebase

# 2. Force remove all old, corrupted, or misnamed files from Git
!git rm -f app.py model_pipeline.py dsn_mart.py dsn_mart_sales_prediction.py 2>/dev/null

# 3. Add the clean script and configuration files to the staging area
!git add streamlit_app.py requirements.txt

# 4. Save and commit your clean updates
!git commit -m "fix: removed embedded terminal commands and standardized entry point"

# 5. Force-push to clear out any old history conflicts on GitHub
!git push origin main --force

