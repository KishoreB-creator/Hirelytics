# 🚀 Hirelytics: AI-Powered SHL Assessment Recommender

Hirelytics is an **AI-driven recommendation system** that intelligently suggests the most relevant **SHL assessments** for a given **job description** or **hiring context**.  
It leverages **semantic embeddings**, **hybrid scoring**, and **Gemini-based reranking** to ensure context-aware, high-quality recommendations in real-time.

---

## 🧠 Features

- 🔍 **Smart Recommendation Engine** — Maps hiring contexts to SHL assessments.  
- 🧩 **Hybrid Ranking Algorithm** — Combines vector similarity (70%) + keyword overlap (30%).  
- 🧠 **LLM-Powered Reranking (Gemini)** — Contextual reranking for nuanced results.  
- ⚡ **Fast & Scalable** — Built with FastAPI and deployed via Docker on Azure.  
- 🌐 **Supports Text & URLs** — Accepts either direct text queries or JD links.  
- 🖤 **Modern UI** — Minimal dark-mode frontend for quick, user-friendly interaction.

---

## 🏗️ System Architecture

| Layer | Technology | Role |
|-------|-------------|------|
| **Frontend** | HTML, CSS, JavaScript | User query input & results display |
| **Backend** | FastAPI + Uvicorn | API serving and model inference |
| **Database** | Supabase (Postgres + pgvector) | Stores SHL assessment embeddings |
| **Model** | SentenceTransformer `all-mpnet-base-v2` | Embedding generator |
| **Reranker** | Google Gemini 2.5 Flash | Contextual reranking |
| **Deployment** | Azure App Service (Docker) | Scalable hosting environment |

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/hirelytics.git
cd hirelytics
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the root directory:
```bash
SUPABASE_URL=<your_supabase_url>
SUPABASE_KEY=<your_supabase_key>
GEMINI_API_KEY=<your_gemini_api_key>
```

### 5. Run Locally
```bash
uvicorn main:app --reload
```

Now open:
```
http://127.0.0.1:8000/docs
```

---

## 🧩 API Endpoints

### **1️⃣ Health Check**
**GET** `/health`
```json
{
  "status": "ok"
}
```

### **2️⃣ Get Recommendations**
**POST** `/recommend`

**Request Body:**
```json
{
  "query": "Hiring a software engineer experienced in Java and teamwork",
  "k": 10
}
```

**Response:**
```json
{
  "recommended_assessments": [
    {
      "name": "Java Developer Aptitude Test",
      "test_type": ["Ability & Aptitude"],
      "duration": "30 min",
      "remote_support": "Yes",
      "description": "Measures analytical thinking and problem-solving...",
      "url": "https://shl.com/test/java"
    }
  ]
}
```

---

## 🚀 Deployment on Azure (Docker)

1. **Build Docker image**
   ```bash
   docker build -t shl-recommender .
   ```

2. **Push to Azure Container Registry**
   ```bash
   az acr login --name <registry-name>
   docker tag shl-recommender <registry-name>.azurecr.io/shl-recommender:latest
   docker push <registry-name>.azurecr.io/shl-recommender:latest
   ```

3. **Deploy Web App**
   ```bash
   az webapp create      --resource-group <resource-group>      --plan <app-service-plan>      --name shl-recommender      --deployment-container-image-name <registry-name>.azurecr.io/shl-recommender:latest
   ```

4. Access the live API at:
   ```
   https://shl-recommender.azurewebsites.net/
   ```

---

## 📈 Performance Summary

| Metric | Before | After Optimization |
|--------|---------|--------------------|
| **Mean Recall@10** | 0.56 | **0.62** |

---

## 💡 Key Optimizations

- Switched from TF-IDF → **Sentence-BERT embeddings**  
- Integrated **Supabase pgvector** for fast semantic retrieval  
- Added **hybrid similarity formula (70/30)**  
- Applied **Gemini summarization + reranking**  
- **Dockerized** FastAPI app for faster cold starts  

---

## 🧭 Future Improvements

- Add **user feedback learning loop** for dynamic tuning  
- Support **multi-language job descriptions**  
- Integrate **analytics dashboard** for usage insights  

---

## 👨‍💻 Author
**Kishore B**  
B.Tech CSE (AI) — 3rd Year  
Hirelytics © 2025

---

## 🏁 License
This project is released under the **MIT License**.
