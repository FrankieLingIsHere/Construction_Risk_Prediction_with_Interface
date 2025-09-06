# Construction Risk Prediction Interface

This project provides an AI-powered interface for construction safety risk prediction, PPE compliance detection, and OSHA guideline integration using Large Language Models (TinyLlama, Gemini).

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/FrankieLingIsHere/Construction_Risk_Prediction_with_Interface.git
cd Construction_Risk_Prediction_with_Interface
```

### 2. Create a Python Virtual Environment

Create and activate a virtual environment:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Interface
```bash
python src/app.py
```

The **Gradio web interface** will launch.  
Follow the instructions on the page to input construction scenarios and view:

- **Risk Predictions**
- **PPE Analysis**
- **OSHA Guideline Recommendations**

---

## Notes

- Make sure you have a **valid Gemini API key** and access to the required model files. Replace your api key in the app.py file so that it can run the interface properly. Get your API Key here: [Google API KEY](https://aistudio.google.com/app/u/3/apikey)
- For best results, use the **provided example scenarios** or detailed construction incident descriptions.  

---

## Troubleshooting

- If you encounter **missing package errors**, ensure your virtual environment is activated and all dependencies are installed.  
- For **GPU acceleration**, ensure **PyTorch** is installed with CUDA support.  
