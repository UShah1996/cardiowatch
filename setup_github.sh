#!/bin/bash
# =============================================================
# CardioWatch - GitHub Project Setup Script
# Run this script once to initialize your project locally
# and push it to GitHub.
# Usage: bash setup_github.sh <your-github-username>
# =============================================================

set -e  # Exit on any error

GITHUB_USERNAME=${1:-"your-username"}
REPO_NAME="cardiowatch"
PYTHON_VERSION="3.10"

echo "============================================="
echo "  CardioWatch GitHub Setup"
echo "  GitHub User: $GITHUB_USERNAME"
echo "============================================="

# ── 1. Check prerequisites ─────────────────────────────────
echo ""
echo "[1/6] Checking prerequisites..."

check_command() {
  if ! command -v "$1" &> /dev/null; then
    echo "  ✗ '$1' not found. Please install it and re-run."
    exit 1
  else
    echo "  ✓ $1 found"
  fi
}

check_command git
check_command python3

# pip may be named pip3 on some systems
if command -v pip &> /dev/null; then
  PIP="pip"
  echo "  ✓ pip found"
elif command -v pip3 &> /dev/null; then
  PIP="pip3"
  echo "  ✓ pip3 found (using as pip)"
else
  echo "  ✗ Neither 'pip' nor 'pip3' found. Please install Python pip and re-run."
  exit 1
fi

# ── 2. Initialize git repo ─────────────────────────────────
echo ""
echo "[2/6] Initializing git repository..."

if [ ! -d ".git" ]; then
  git init
  echo "  ✓ Git repository initialized"
else
  echo "  ✓ Git repository already exists"
fi

git config user.name "$GITHUB_USERNAME"
echo "  ✓ Git username set to '$GITHUB_USERNAME'"

# ── 3. Create virtual environment ─────────────────────────
echo ""
echo "[3/6] Setting up Python virtual environment..."

if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "  ✓ Virtual environment created"
else
  echo "  ✓ Virtual environment already exists"
fi

source venv/bin/activate
$PIP install --upgrade pip -q
$PIP install -r requirements.txt -q
echo "  ✓ Dependencies installed"

# ── 4. Stage all files ────────────────────────────────────
echo ""
echo "[4/6] Staging project files..."

git add .
git commit -m "feat: initial CardioWatch project scaffold

- Project structure for ECG + clinical data pipeline
- CNN-LSTM model architecture skeleton
- Preprocessing modules (band-pass filter, SMOTE, windowing)
- Streamlit dashboard scaffold
- Evaluation utilities (AUC-ROC, recall, SHAP)
- Configs for reproducible experiments
- Notebooks for EDA and model exploration"

echo "  ✓ Initial commit created"

# ── 5. Connect to GitHub ──────────────────────────────────
echo ""
echo "[5/6] Connecting to GitHub remote..."

REMOTE_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

if git remote get-url origin &> /dev/null; then
  git remote set-url origin "$REMOTE_URL"
  echo "  ✓ Remote 'origin' updated to $REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
  echo "  ✓ Remote 'origin' set to $REMOTE_URL"
fi

# ── 6. Push to GitHub ─────────────────────────────────────
echo ""
echo "[6/6] Pushing to GitHub..."
echo ""
echo "  ⚠  BEFORE PUSHING — make sure you have:"
echo "     1. Created the repo at: https://github.com/new"
echo "        Repo name: $REPO_NAME"
echo "        Visibility: Private (recommended for ML projects)"
echo "        Do NOT initialize with README (we already have one)"
echo ""
echo "     2. Authenticated git (choose one):"
echo "        • SSH key  → git remote set-url origin git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
echo "        • HTTPS    → use a Personal Access Token as password"
echo "          Create at: https://github.com/settings/tokens"
echo ""
read -p "  Ready to push? (y/n): " CONFIRM

if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
  git branch -M main
  git push -u origin main
  echo ""
  echo "  ✓ Code pushed to: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
else
  echo ""
  echo "  Skipped push. When ready, run:"
  echo "    git branch -M main && git push -u origin main"
fi

echo ""
echo "============================================="
echo "  Setup complete! Next steps:"
echo "  1. Activate venv:  source venv/bin/activate"
echo "  2. Download data:  python src/preprocessing/download_data.py"
echo "  3. Run dashboard:  streamlit run src/dashboard/app.py"
echo "============================================="
