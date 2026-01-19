## GenAI Food Fabrication

### Setup

#### Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

#### Frontend
cd frontend
npm install
npm run dev

### Usage
Open http://localhost:3000
Enter a prompt and click Run
