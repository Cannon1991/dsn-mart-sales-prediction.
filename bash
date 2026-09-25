echo -e "streamlit>=1.35.0\npandas>=2.0.0\nnumpy>=1.24.0\nscikit-learn>=1.3.0" > requirements.txt
echo -e "train.csv\ntest.csv\nsubmission.csv\n.ipynb_checkpoints/\n__pycache__/" > .gitignore
# 1. Initialize an empty local git tracking directory repository
git init

# 2. Add all your source scripts and requirement files safely
git add model_pipeline.py app.py requirements.txt .gitignore

# 3. Save your code staging branch with a commit marker message
git commit -m "feat: finalized dsn mart sales tracking model and clean streamlit frontend pipeline ready for launch"

# 4. Set your primary branch track naming standard designation to main
git branch -M main

# 5. Link your local directory layout directly to your official GitHub repo space
# 📊 NOTE: Replace 'YOUR_GITHUB_USERNAME' with your actual public GitHub profile handle
git remote add origin https://github.com

# 6. Securely push and deploy your project files live to the master cloud branch
git push -u origin main
# 1. Rename your local user interface script file to match the default port standard
mv app.py streamlit_app.py

# 2. Stage the file removal and addition updates into git tracking memory
git add streamlit_app.py
git rm app.py

# 3. Save the branch updates with a tracking comment marker
git commit -m "fix: updated dashboard entry script name to match standard cloud portal defaults"

# 4. Push the structural file updates directly up to your GitHub repository
git push origin main
