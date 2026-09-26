# 1. Fetch online changes to stay synchronized with the cloud branch
!git pull origin main --rebase

# 2. Tell git to entirely drop the old files causing syntax errors
!git rm -f app.py model_pipeline.py dsn_mart.py dsn_mart_sales_prediction.py 2>/dev/null

# 3. Add the clean script and configuration files to the staging area
!git add streamlit_app.py requirements.txt

# 4. Commit and force-push the fresh state straight to GitHub
!git commit -m "fix: purged script files with raw terminal syntax errors"
!git push origin main --force

