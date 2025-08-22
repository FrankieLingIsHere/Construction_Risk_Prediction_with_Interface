import torch
import re
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import gc

# === Gemini setup ===
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()

# === Configuration ===
MODEL_PATHS = [
    "FrAnKu34t23/Construction_Risk_Prediction_TinyLlama_M3_new",      # Most reliable
    "FrAnKu34t23/Construction_Risk_Prediction_TinyLlama_M1_new",      # Medium reliable
    "FrAnKu34t23/Construction_Risk_Prediction_TinyLlama_M2_latest"   # Least reliable
]
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def infer_injury_degree_from_scenario(scenario_text):
    """
    Intelligently infer injury degree from scenario context when models fail
    This ensures we NEVER return 'Unknown' injury degree
    """
    scenario_lower = scenario_text.lower()

    # High risk indicators (likely fatal or could cause death)
    high_risk_indicators = [
        "explosion", "fire", "chemical", "toxic", "fall", "height", "electrocuted",
        "electric shock", "crush", "caught between", "machinery", "equipment failure",
        "respiratory distress", "cardiac", "stroke", "overdose", "violence", "shot",
        "severe", "fatal", "unconscious", "collapsed", "bleeding", "head injury"
    ]

    # Medium risk indicators (likely hospitalization)
    medium_risk_indicators = [
        "fracture", "broken", "burn", "laceration", "cut", "sprain", "strain",
        "injured", "hurt", "pain", "medical attention", "hospital", "stitches",
        "bruise", "swelling", "dizzy", "nausea", "respiratory", "breathing"
    ]

    # Low risk indicators (minor injury, no hospitalization)
    low_risk_indicators = [
        "minor", "scratch", "small cut", "bandage", "first aid", "bruised",
        "sore", "tired", "headache", "eye irritation", "skin irritation"
    ]

    # Count matches for each category
    high_count = sum(1 for word in high_risk_indicators if word in scenario_lower)
    medium_count = sum(1 for word in medium_risk_indicators if word in scenario_lower)
    low_count = sum(1 for word in low_risk_indicators if word in scenario_lower)

    # Decision logic
    if high_count > 0:
        return "High"
    elif medium_count > 0:
        return "Medium"
    elif low_count > 0:
        return "Low"
    else:
        # Default fallback - if we can't determine, assume Medium as safe middle ground
        return "Medium"

def is_accident_scenario(scenario: str) -> bool:
    # Load few-shot PPE samples
    try:
        with open('ppe_fewshot_samples.txt', 'r', encoding='utf-8') as f:
            fewshot_examples = f.read().strip()
    except Exception:
        fewshot_examples = ''
    prompt = f"""You are an occupational safety assistant.

{fewshot_examples}
Scenario:
{scenario.strip()}

Given the above, determine if this scenario either **has already caused** or **could cause** a workplace accident or injury.
Respond ONLY with one word: "Yes" or "No"."""
    try:
        response = chat.send_message(prompt)
        result = response.text.strip().lower()
        return result.startswith("yes")
    except Exception as e:
        print(f"[is_accident_scenario] Gemini check failed: {e}")
        return True  # fallback to processing if uncertain

# === Load RAG Index & Embeddings ===
index = faiss.read_index("Osha_GuidelineLLM/osha_rag_index.faiss")
with open("Osha_GuidelineLLM/osha_rag_metadata.json", "r", encoding="utf-8") as f:
    chunk_id_to_metadata = json.load(f)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_osha_guidelines(query, top_k=5):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    results = []
    for idx in I[0]:
        meta = chunk_id_to_metadata[str(idx)]
        results.append({
            "source_file": meta["source_file"],
            "chunk_text": meta["chunk_text"],
            "chunk_id": meta["chunk_id"]
        })
    return results

def generate_osha_explanation_with_gemini(incident_description, top_chunks):
    context = "\n\n".join(
        [f"From {c['source_file']} (Chunk {c['chunk_id']}):\n{c['chunk_text']}" for c in top_chunks]
    )
    prompt = f"""
You are a construction safety expert.

A construction incident occurred with the following description:

"{incident_description}"

Below are relevant OSHA guideline excerpts:

{context}

Please explain in short point form:
1. What OSHA rules may have been violated?
2. Why the situation was unsafe?
3. What should have been done to prevent it?

Provide your explanation in clear, professional language.
"""
    response = chat.send_message(prompt)
    return response.text

# === 💾 MEMORY EFFICIENT MODEL LOADING ===
print("🔄 Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, trust_remote_code=True).to("cpu").eval()
print("✅ Base model loaded!")

# Keep track of current adapter
current_adapter = None
current_adapter_path = None

def load_adapter_on_demand(adapter_path, model_num):
    """Load adapter only when needed, unload previous one"""
    global current_adapter, current_adapter_path

    # If this adapter is already loaded, return it
    if current_adapter_path == adapter_path and current_adapter is not None:
        return current_adapter

    # Unload previous adapter if exists
    if current_adapter is not None:
        print(f"🗑️ Unloading previous adapter: {current_adapter_path}")
        del current_adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load new adapter
    print(f"🔄 Loading adapter {model_num}: {adapter_path}")
    current_adapter = PeftModel.from_pretrained(base_model, adapter_path).to("cpu").eval()
    current_adapter_path = adapter_path
    print(f"✅ Adapter {model_num} loaded!")

    return current_adapter

def format_input(scenario_text):
    scenario = scenario_text.strip()
    return f"Based on the situation, predict potential hazards and injuries. {scenario}\nOutput:\n"

def extract_json_only(text):
    try:
        # Find all JSON objects in the text
        json_candidates = re.findall(r"\{[^{}]*\}", text)
        required_fields = ["Hazards", "Cause of Accident", "Degree of Injury"]
        for raw_json in json_candidates:
            cleaned_json = raw_json.replace(""", '"').replace(""", '"') \
                                   .replace("'", '"').replace("'", '"') \
                                   .replace("''", '"').replace("†", "")
            cleaned_json = re.sub(r",\s*}", "}", cleaned_json)
            cleaned_json = re.sub(r",\s*]", "]", cleaned_json)
            try:
                parsed = json.loads(cleaned_json)
                # Check for required fields
                if all(field in parsed for field in required_fields):
                    return parsed
            except Exception:
                continue
        return None
    except Exception as e:
        print(f"[extract_json_only] Failed to parse JSON: {e}")
        return None

def generate_single_model_output(adapter_path, model_num, prompt, max_length=300, temperature=0.7):
    """Generate output using on-demand adapter loading"""
    # Load adapter on demand
    adapter = load_adapter_on_demand(adapter_path, model_num)

    # Set unique seed for each model to ensure different outputs
    torch.manual_seed(42 + model_num * 17)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cpu")
    with torch.no_grad():
        output = adapter.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            num_return_sequences=1
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return result

def verify_single_model_quality_with_gemini(raw_output, scenario_text, model_num, debug_info):
    """
    🤖 GEMINI-POWERED QUALITY CHECK - Much more intelligent than formula-based
    """
    debug_info.append(f"\n🤖 GEMINI QUALITY CHECK - Model {model_num}:")
    debug_info.append(f"   🎯 Using Gemini to evaluate model output quality")

    json_output = extract_json_only(raw_output)

    # Basic structure check (still needed)
    if json_output is None:
        debug_info.append("   ❌ BASIC CHECK: Invalid JSON format")
        return False, "Poor", "Invalid JSON format", None

    required_fields = ["Hazards", "Cause of Accident", "Degree of Injury"]
    missing_fields = [field for field in required_fields if field not in json_output or not json_output[field]]
    if missing_fields:
        debug_info.append(f"   ❌ BASIC CHECK: Missing fields: {', '.join(missing_fields)}")
        return False, "Poor", f"Missing fields: {', '.join(missing_fields)}", None

    debug_info.append("   ✅ BASIC CHECK: JSON format and required fields OK")

    # 🤖 GEMINI COMPREHENSIVE EVALUATION
    gemini_prompt = f"""You are a construction safety expert evaluating an AI model's prediction quality.

SCENARIO: "{scenario_text}"

MODEL {model_num} PREDICTION:
{json.dumps(json_output, indent=2)}

Please evaluate this prediction on multiple criteria:

1. **CAUSE RELEVANCE**: Does the "Cause of Accident" make sense for this scenario?
   - Is it specific enough (not generic like "lack of safety measures")?
   - Does it match the actual accident type described?
   - Is it actionable for prevention?

2. **HAZARD ACCURACY**: Are the predicted hazards relevant and comprehensive?
   - Do they relate to the scenario?
   - Are important hazards missing?

3. **INJURY SEVERITY**: Is the "Degree of Injury" appropriate?
   - Does it match the scenario's potential severity?
   - Is it realistic?

4. **OVERALL QUALITY**: Is this prediction good enough to use without additional models?

Rate the overall quality on a scale of 1-5:
- 1 = Poor (wrong, irrelevant, or too generic)
- 2 = Fair (somewhat relevant but significant issues)
- 3 = Good (decent but could be better)
- 4 = Very Good (accurate and specific - good enough to stop cascade)
- 5 = Excellent (highly accurate and comprehensive)

Respond in this EXACT format:
Score: [1-5]
Quality: [Poor/Fair/Good/Very Good/Excellent]
Cause Assessment: [Specific/Generic/Wrong/Good]
Issues: [specific problems, or "None"]
Recommendation: [Continue Cascade/Stop Cascade]"""

    try:
        response = chat.send_message(gemini_prompt)
        response_text = response.text.strip()
        debug_info.append(f"   🤖 GEMINI RESPONSE: {response_text[:200]}...")

        # Parse Gemini response
        score_match = re.search(r"Score:\s*([1-5])", response_text)
        quality_match = re.search(r"Quality:\s*(Poor|Fair|Good|Very Good|Excellent)", response_text)
        cause_match = re.search(r"Cause Assessment:\s*(Specific|Generic|Wrong|Good)", response_text)
        issues_match = re.search(r"Issues:\s*(.*?)(?=\n|$)", response_text, re.DOTALL)
        recommendation_match = re.search(r"Recommendation:\s*(Continue Cascade|Stop Cascade)", response_text)

        if not all([score_match, quality_match]):
            debug_info.append(f"   ❌ GEMINI PARSE ERROR")
            return False, "Unknown", "Failed to parse Gemini assessment", json_output

        score = int(score_match.group(1)) if score_match else -1
        quality = quality_match.group(1) if quality_match else "Unknown"
        cause_assessment = cause_match.group(1) if cause_match else "Unknown"
        issues = issues_match.group(1).strip() if issues_match else "Unknown"
        recommendation = recommendation_match.group(1) if recommendation_match else "Continue Cascade"

        debug_info.append(f"   📊 GEMINI RESULTS:")
        debug_info.append(f"      Score: {score}/5")
        debug_info.append(f"      Quality: {quality}")
        debug_info.append(f"      Cause: {cause_assessment}")
        debug_info.append(f"      Issues: {issues}")
        debug_info.append(f"      Recommendation: {recommendation}")

        # Decision logic based on Gemini's comprehensive evaluation
        is_good_enough = score >= 4 and recommendation == "Stop Cascade"

        if is_good_enough:
            debug_info.append(f"   ✅ CASCADE DECISION: STOP - Model {model_num} is satisfactory!")
        else:
            debug_info.append(f"   🔄 CASCADE DECISION: CONTINUE - Model {model_num} needs improvement")

        return is_good_enough, quality, f"{cause_assessment} cause, Issues: {issues}", json_output
    except Exception as e:
        debug_info.append(f"   ❌ GEMINI ERROR: {str(e)}")
        debug_info.append("   🔄 FALLBACK: Continuing cascade due to evaluation failure")
        return False, "Unknown", f"Gemini evaluation failed: {e}", json_output

def analyze_with_gemini_final_integration(raw_outputs, scenario_text, debug_info):
    """
    🎯 FINAL GEMINI INTEGRATION: Process all generated outputs for final result
    🎯 ENHANCED: Separate cause extraction from risk assessment
    """
    debug_info.append(f"\n🤖 FINAL GEMINI INTEGRATION:")
    debug_info.append(f"   🎯 Integrating {len(raw_outputs)} generated model outputs")
    debug_info.append(f"   🧠 Gemini will create separate cause analysis and risk assessment")

    prompt = (
        "You are a safety incident analyst.\n"
        "Below are model outputs from different AI models for the same construction incident. "
        "Some models may have been skipped if an earlier model was good enough. "
        "Your task is to:\n"
        "1. Extract the most accurate **cause of accident** from the model outputs (focus ONLY on what caused the incident)\n"
        "2. Separately assess the **injury risk** level based on the scenario and model predictions\n\n"
        f"SCENARIO: {scenario_text}\n\n"
    )

    for i, output in enumerate(raw_outputs):
        prompt += f"Model {i+1} Output:\n{output}\n\n"

    prompt += (
        "INSTRUCTIONS:\n"
        "- For CAUSE: Extract only the most specific and relevant cause from the models' 'Cause of Accident' fields\n"
        "- If model causes are poor/ambiguous, infer the correct cause from the scenario context\n"
        "- For RISK: Consider scenario severity, potential injuries, and models' 'Degree of Injury' predictions\n"
        "- Risk levels: Low (minor/first aid), Medium (hospitalization), High (life-threatening/fatal)\n"
        "- NEVER use 'Unknown' for risk - always choose High, Medium, or Low\n\n"
        "Respond in this EXACT format:\n"
        "Cause: <specific cause of the accident in natural language>\n"
        "Risk: <High/Medium/Low>"
    )

    try:
        response = chat.send_message(prompt)
        result = response.text.strip()

        # Validate risk is present
        if not re.search(r"Risk:\s*(High|Medium|Low)", result, re.IGNORECASE):
            debug_info.append(f"   ⚠️ RISK MISSING: Adding fallback risk...")
            fallback_risk = infer_injury_degree_from_scenario(scenario_text)
            result += f"\nRisk: {fallback_risk}"
            debug_info.append(f"   ✅ FALLBACK RISK ADDED: {fallback_risk}")

        debug_info.append(f"   ✅ INTEGRATION SUCCESS: Separate cause and risk extracted")
        return result
    except Exception as e:
        debug_info.append(f"   ❌ GEMINI ERROR: {str(e)}")
        fallback_risk = infer_injury_degree_from_scenario(scenario_text)
        return f"Cause: Workplace incident as described in scenario.\nRisk: {fallback_risk}"

def extract_cause_and_risk_separately(gemini_response, scenario_text):
    """
    🎯 ENHANCED: Extract cause and risk separately from Gemini response
    """
    # Extract cause (everything after "Cause:" and before "Risk:")
    cause_match = re.search(r"Cause:\s*(.*?)(?=\nRisk:|\nrisk:|\n|$)", gemini_response, re.IGNORECASE | re.DOTALL)
    cause = cause_match.group(1).strip() if cause_match else "Workplace incident occurred as described"

    # Extract risk level
    risk_match = re.search(r"Risk:\s*(High|Medium|Low)", gemini_response, re.IGNORECASE)
    risk = risk_match.group(1).capitalize() if risk_match else infer_injury_degree_from_scenario(scenario_text)

    return cause, risk

def generate_prediction_cascade_with_gemini_evaluation(scenario_text, max_len, temperature, skip_check):
    import concurrent.futures
    """
    🎯 ENHANCED CASCADE with Gemini-based quality evaluation
    """
    debug_info = []
    debug_info.append("🚀 ENHANCED CASCADE WITH GEMINI EVALUATION")
    debug_info.append(f"📋 Configuration: max_len={max_len}, temperature={temperature}")
    debug_info.append(f"🤖 Quality Check: Gemini-powered (no formula-based verification)")
    import time
    timings = {}
    overall_start = time.time()

    try:
        if not scenario_text.strip():
            return "Please enter a scenario", "", "\n".join(debug_info), ""

        # Optional accident scenario validation
        t0 = time.time()
        if not skip_check:
            debug_info.append("\n🔍 SCENARIO VALIDATION:")
            if not is_accident_scenario(scenario_text):
                debug_info.append("   ❌ Not an accident scenario")
                timings['scenario_validation'] = time.time() - t0
                return (
                    "The scenario doesn't seem to cause any workplace accident.",
                    "Low",
                    "\n".join(debug_info) + "\n\nModel skipped — scenario deemed not accident-related.",
                    "OSHA analysis skipped — no accident detected."
                )
            else:
                debug_info.append("   ✅ Valid accident scenario detected")
            timings['scenario_validation'] = time.time() - t0
        else:
            debug_info.append("\n⚠️ SCENARIO VALIDATION: SKIPPED (as requested)")
            timings['scenario_validation'] = time.time() - t0

        prompt = format_input(scenario_text)

        timings_model = []
        raw_outputs = []
        gemini_results = []
        timings_eval = []
        cascade_stopped_at = None
        models_tested = 0
        t1 = time.time()
        for i, adapter_path in enumerate(MODEL_PATHS, 1):
            t_model_start = time.time()
            result = generate_single_model_output(adapter_path, i, prompt, max_len, temperature)
            raw_outputs.append(result)
            timings_model.append(time.time() - t_model_start)
            # Gemini evaluation for this output
            t_eval_start = time.time()
            gemini_eval = None
            if result is not None:
                try:
                    gemini_eval = verify_single_model_quality_with_gemini(result, scenario_text, i, debug_info)
                    gemini_results.append(gemini_eval)
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    error_msg = f"❌ ERROR in Gemini evaluation for model {i}: {str(e)}\nTraceback:\n{error_trace}"
                    gemini_results.append(None)
                    debug_info.append(error_msg)
            else:
                gemini_results.append(None)
                timings_eval.append(0.0)
                continue
            timings_eval.append(time.time() - t_eval_start)
            # Check if Gemini says this model is satisfactory
            if gemini_eval and isinstance(gemini_eval, tuple):
                is_satisfactory, _, _, _ = gemini_eval
                if is_satisfactory:
                    cascade_stopped_at = i
                    models_tested = i
                    debug_info.append(f"\n✅ GEMINI APPROVED: Model {i} is satisfactory!")
                    debug_info.append(f"🎯 Models tested: {models_tested}/{len(MODEL_PATHS)}")
                    debug_info.append(f"💾 Efficiency: Saved {len(MODEL_PATHS) - models_tested} model runs")
                    break
        if cascade_stopped_at is None:
            models_tested = len(MODEL_PATHS)
            debug_info.append(f"\n⚠️ CASCADE COMPLETED: No model met Gemini's standards")
            debug_info.append(f"🎯 All {models_tested} models tested")
        for i, tgen in enumerate(timings_model, 1):
            timings[f'model_{i}_generation'] = tgen
        for i, teval in enumerate(timings_eval, 1):
            timings[f'model_{i}_gemini_eval'] = teval
        timings['cascade_phase'] = time.time() - t1

        # Cascade decision logic (sequentially check results)
        for i, result in enumerate(gemini_results, 1):
            if raw_outputs[i-1] is None:
                debug_info.append(f"\n🔄 TESTING MODEL {i} WITH GEMINI...")
                debug_info.append(f"   Model path: {MODEL_PATHS[i-1]}")
                debug_info.append(f"   📝 Response generated (no output due to error, Gemini evaluation skipped)")
                continue
            if result is None:
                debug_info.append(f"\n🔄 TESTING MODEL {i} WITH GEMINI...")
                debug_info.append(f"   Model path: {MODEL_PATHS[i-1]}")
                debug_info.append(f"   📝 Response generated (Gemini evaluation failed)")
                continue
            try:
                is_satisfactory, quality_score, issues, json_output = result
            except Exception:
                debug_info.append(f"\n🔄 TESTING MODEL {i} WITH GEMINI...")
                debug_info.append(f"   Model path: {MODEL_PATHS[i-1]}")
                debug_info.append(f"   📝 Response generated (invalid result)")
                continue
                debug_info.append(f"\n🔄 TESTING MODEL {i} WITH GEMINI...")
                debug_info.append(f"   Model path: {MODEL_PATHS[i-1]}")
                debug_info.append(f"   📝 Response generated ({len(raw_outputs[i-1])} chars)")
                if is_satisfactory:
                    cascade_stopped_at = i
                    models_tested = i
                    debug_info.append(f"\n✅ GEMINI APPROVED: Model {i} is satisfactory!")
                    debug_info.append(f"🎯 Models tested: {models_tested}/{len(MODEL_PATHS)}")
                    debug_info.append(f"💾 Efficiency: Saved {len(MODEL_PATHS) - models_tested} model runs")
                    break
                else:
                    debug_info.append(f"\n❌ GEMINI REJECTED: Model {i} - {quality_score}")
                    debug_info.append(f"🔄 Continuing to next model...")
            if cascade_stopped_at is None:
                models_tested = len(MODEL_PATHS)
                debug_info.append(f"\n⚠️ CASCADE COMPLETED: No model met Gemini's standards")
                debug_info.append(f"🎯 All {models_tested} models tested")

            # Improved integration: show traceable model outputs and highlight 'Good' responses
            t2 = time.time()
            debug_info.append(f"\n🎯 GEMINI INTEGRATION")
            debug_info.append(f"📊 Integrating {len(raw_outputs)} outputs for final result...")

            integration_summary = []
            best_score = -1
            best_model_idx = None
            best_output = None
            gemini_assessments = []
            for i, (output, gemini_result) in enumerate(zip(raw_outputs, gemini_results), 1):
                integration_summary.append(f"=== MODEL {i} OUTPUT ===\n{output if output else 'No output (error)'}")
                score, quality, cause_assessment = -1, "Unknown", "Unknown"
                # Always use parsed Gemini result if available
                if gemini_result is not None and isinstance(gemini_result, tuple):
                    score = gemini_result[0] if isinstance(gemini_result[0], int) else -1
                    quality = gemini_result[1] if isinstance(gemini_result[1], str) else "Unknown"
                    # Try to extract cause assessment from the string
                    cause_assessment_match = re.search(r"(Specific|Generic|Wrong|Good)", str(gemini_result))
                    cause_assessment = cause_assessment_match.group(1) if cause_assessment_match else "Unknown"
                integration_summary.append(f"Gemini Assessment: Quality={quality}, Cause Assessment={cause_assessment}")
                gemini_assessments.append((score, quality, cause_assessment))
                if score > best_score:
                    best_score = score
                    best_model_idx = i
                    best_output = output

            final_cause = None
            final_risk = None
            cause_flagged = False
            highlight_shown = False
            # Use best Gemini-rated output if available
            if best_score == 3 and best_output and best_model_idx is not None:
                integration_summary.append(f"\n⭐ Highlight: Model {best_model_idx} received a 'Good' score from Gemini. Its output is prioritized in the final result.")
                highlight_shown = True
                json_data = extract_json_only(best_output)
                if json_data is not None:
                    final_cause = json_data.get("Cause of Accident")
                    final_risk = json_data.get("Degree of Injury")
                    if best_model_idx-1 < len(gemini_assessments):
                        _, _, cause_assessment = gemini_assessments[best_model_idx-1]
                        if cause_assessment in ["Wrong", "Generic"] or not final_cause or len(str(final_cause)) < 10:
                            cause_flagged = True
            elif best_score > 3 and best_output and best_model_idx is not None:
                integration_summary.append(f"\n⭐ Highlight: Model {best_model_idx} received a 'Very Good' or 'Excellent' score from Gemini. Its output is prioritized in the final result.")
                highlight_shown = True
                json_data = extract_json_only(best_output)
                if json_data is not None:
                    final_cause = json_data.get("Cause of Accident")
                    final_risk = json_data.get("Degree of Injury")
                    if best_model_idx-1 < len(gemini_assessments):
                        _, _, cause_assessment = gemini_assessments[best_model_idx-1]
                        if cause_assessment in ["Wrong", "Generic"] or not final_cause or len(str(final_cause)) < 10:
                            cause_flagged = True
            # Fallback: Use first valid model output if no highlight
            if not highlight_shown:
                for idx, output in enumerate(raw_outputs):
                    json_data = extract_json_only(output)
                    if json_data is not None:
                        final_cause = json_data.get("Cause of Accident")
                        final_risk = json_data.get("Degree of Injury")
                        if idx < len(gemini_assessments):
                            _, _, cause_assessment = gemini_assessments[idx]
                            if cause_assessment in ["Wrong", "Generic"] or not final_cause or len(str(final_cause)) < 10:
                                cause_flagged = True
                                integration_summary.append("\nNo model output was rated 'Good' or better. Using first valid output for final result.")
                        break
                if not final_cause:
                    integration_summary.append("\nNo valid model output found. Cause set to scenario summary.")
                    final_cause = f"Incident: {scenario_text[:80]}..."
                    cause_flagged = True
                if not final_risk:
                    final_risk = infer_injury_degree_from_scenario(scenario_text)
                    integration_summary.append(f"   🛡️ FINAL SAFETY CHECK: Risk set to {final_risk}")



            # Always process the final cause into natural language using Gemini
            if final_cause:
                # Use Gemini to rephrase the cause in natural language
                cause_prompt = (

                    "You are a safety expert. Given the scenario and the model's cause of accident, "
                    "write a single, clear sentence describing the actual cause of the accident for a safety report. "
                    "If the model's cause is incorrect or irrelevant, ignore it and infer the correct cause from the scenario context. "

                    "Do not mention or critique the model's original cause. "
                    "Respond with only the actual cause in natural language."
                    f"\n\nScenario: {scenario_text}\n\nModel Cause: {final_cause}\n"

                )

                try:
                    cause_response = chat.send_message(cause_prompt)
                    final_cause_natural = cause_response.text.strip() if hasattr(cause_response, 'text') else str(cause_response).strip()
                    if final_cause_natural:
                        final_cause_natural = final_cause_natural.split(';')[0].strip()
                        final_cause = final_cause_natural
                        integration_summary.append("   📝 Final cause processed into natural language by Gemini.")
                except Exception:
                    integration_summary.append("   ⚠️ Failed to process cause into natural language, using fallback.")

            timings['integration_phase'] = time.time() - t2

            integration_summary.append(f"\n📝 FINAL RESULT:")
            integration_summary.append(f"   📝 Cause: {final_cause}")
            integration_summary.append(f"   📊 Risk: {final_risk}")
            integration_summary.append(f"   ⚡ Efficiency: {models_tested}/{len(MODEL_PATHS)} models used")
            integration_summary.append(f"   🤖 Quality Control: Gemini-powered evaluation")

            # OSHA integration
            t3 = time.time()
            integration_summary.append(f"\n🔍 OSHA ANALYSIS:")
            top_chunks = []
            osha_explanation = ""
            try:
                top_chunks = search_osha_guidelines(scenario_text, top_k=3)
                osha_explanation = generate_osha_explanation_with_gemini(scenario_text, top_chunks)
                timings['osha_analysis'] = time.time() - t3
                integration_summary.append(f"📚 OSHA guidelines found: {len(top_chunks)} relevant chunks")
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                integration_summary.append(f"❌ OSHA analysis error: {str(e)}")
                integration_summary.append(f"📋 Full traceback:\n{error_trace}")
                osha_explanation = "Error"
            # Append all debug logs to the integration summary for full traceability
            integration_summary.append("\n=== DEBUG LOGS ===")
            integration_summary.extend(debug_info)
            return final_cause, final_risk, "\n\n".join(integration_summary), osha_explanation

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        debug_info.append(f"\n❌ CRITICAL ERROR: {str(e)}")
        debug_info.append(f"📋 Full traceback:\n{error_trace}")

        safe_injury_risk = infer_injury_degree_from_scenario(scenario_text) if scenario_text.strip() else "Medium"
        return "Error processing scenario", safe_injury_risk, "\n".join(debug_info), "Error"

def create_interface():
    with gr.Blocks(title="Construction Risk Predictor") as demo:
        gr.Markdown("## 🏗️ Construction Safety Risk Predictor")

        with gr.Row():
            with gr.Column(scale=1):
                skip_check = gr.Checkbox(label="⚠️ Skip scenario validation (for testing)", value=False)
                scenario_input = gr.Textbox(
                    lines=6,
                    label="Describe the construction scenario",
                    placeholder="Enter a detailed description of the construction scenario..."
                )

            with gr.Column(scale=1):
                cause_output = gr.Textbox(
                    label="📝 Cause of Accident",
                    lines=5,
                    placeholder="The specific cause will appear here..."
                )
                degree_output = gr.Textbox(
                    label="📊 Injury Risk Level",
                    lines=2,
                    placeholder="Risk level (High/Medium/Low) will appear here..."
                )

        with gr.Accordion("📘 OSHA Guidelines & Safety Recommendations", open=False):
            osha_output = gr.Textbox(
                label="OSHA Compliance Analysis",
                lines=25,
                show_copy_button=True,
                placeholder="OSHA guidelines and safety recommendations will appear here..."
            )

        with gr.Accordion("🔍 Detailed Analysis & Model Performance", open=False):
            raw_output = gr.Textbox(
                label="Complete Analysis Process",
                lines=35,
                show_copy_button=True,
                placeholder="Detailed model outputs and analysis process will appear here..."
            )
        gr.Markdown("### 📋 Example Scenarios")
        with gr.Row():
            ex1 = gr.Button("🧪 Chemical Exposure", size="sm")
            ex2 = gr.Button("🪜 Fall Hazard", size="sm")
        with gr.Row():
            ex3 = gr.Button("⚙️ Equipment Malfunction", size="sm")
            ex4 = gr.Button("🔥 Fire Incident", size="sm")

        with gr.Row():
            temperature = gr.Slider(0.1, 1.0, 0.7, label="🎨 Model creativity")
            max_len = gr.Slider(50, 300, 150, label="📏 Response length")

        submit_btn = gr.Button("🔍 Analyze Safety Risk", variant="primary", size="lg")

        submit_btn.click(
            fn=generate_prediction_cascade_with_gemini_evaluation,
            inputs=[scenario_input, max_len, temperature, skip_check],
            outputs=[cause_output, degree_output, raw_output, osha_output]
        )

        # Example scenario buttons
        ex1.click(
            fn=lambda: "An employee was working with chemical solvents in a poorly ventilated area without proper respiratory protection.",
            outputs=scenario_input
        )
        ex2.click(
            fn=lambda: "A construction worker was installing roofing materials on a steep slope without proper fall protection equipment. The worker lost footing on wet materials and fell from a height of 20 feet.",
            outputs=scenario_input
        )
        ex3.click(
            fn=lambda: "During routine maintenance, a hydraulic press malfunctioned due to worn seals.",
            outputs=scenario_input
        )
        ex4.click(
            fn=lambda: "While welding in an area with flammable materials, proper fire safety protocols were not followed. Sparks from the welding operation ignited nearby combustible materials causing a flash fire.",
            outputs=scenario_input
        )

    return demo

if __name__ == "__main__":
    app = create_interface()

    app.launch(server_name="0.0.0.0", share=True)
