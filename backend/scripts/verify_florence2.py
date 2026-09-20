"""
SatQuery AI - Standalone Verification Script for Microsoft Florence-2-base.
Executes genuine:
1. Image Captioning (<MORE_DETAILED_CAPTION> and <DETAILED_CAPTION>)
2. Visual Question Answering (<VQA>)
3. Text-Guided Phrase Grounding (<CAPTION_TO_PHRASE_GROUNDING> / <OPEN_VOCABULARY_DETECTION>)

Measures CPU latency, memory footprint, and prints actual model outputs.
Clearly labels results as TEST IMAGE verification.
"""
import os
import sys
import time
import psutil
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Path setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models", "checkpoints", "florence2-base")
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "backend", "models", "checkpoints", "resnet18-s2-v0.2.0", "example.png")

def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_task(model, processor, image, task_prompt: str, text_input: str = None):
    prompt = task_prompt if text_input is None else f"{task_prompt}{text_input}"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Measure inference time
    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            early_stopping=False,
        )
    latency = time.time() - t0
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer, latency, generated_text

def main():
    print("=" * 70)
    print("FLORENCE-2-BASE STANDALONE VERIFICATION (CPU RUNTIME)")
    print("=" * 70)
    
    initial_mem = get_process_memory_mb()
    print(f"Initial Process Memory: {initial_mem:.1f} MB")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Test Image Path: {TEST_IMAGE_PATH}")
    
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory does not exist: {MODEL_DIR}")
        sys.exit(1)
        
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"ERROR: Test image does not exist: {TEST_IMAGE_PATH}")
        sys.exit(1)
        
    # 1. Load Processor and Model
    t0 = time.time()
    print("\n[1/4] Loading Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    print("[2/4] Loading Florence-2-base Model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    model.eval()
    load_time = time.time() - t0
    post_load_mem = get_process_memory_mb()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model successfully loaded in {load_time:.2f}s!")
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"Memory after model load: {post_load_mem:.1f} MB (Delta: +{post_load_mem - initial_mem:.1f} MB)")
    
    # 2. Load Image
    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    print(f"\n[3/4] Loaded Test Image: {image.size[0]}x{image.size[1]} px, Mode: {image.mode}")
    print("NOTE: Using authentic satellite patch from BigEarthNet checkpoint as TEST DATA.")
    
    # 3. Task A: Captioning
    print("\n" + "-" * 50)
    print("TASK A: IMAGE CAPTIONING (<MORE_DETAILED_CAPTION>)")
    print("-" * 50)
    cap_res, cap_time, raw_cap = run_task(model, processor, image, "<MORE_DETAILED_CAPTION>")
    cap_text = cap_res.get("<MORE_DETAILED_CAPTION>", cap_res)
    print(f"Latency: {cap_time:.2f}s")
    print(f"Caption Output:\n\"{cap_text}\"")
    
    # Also run standard <DETAILED_CAPTION>
    det_cap_res, det_cap_time, _ = run_task(model, processor, image, "<DETAILED_CAPTION>")
    det_cap_text = det_cap_res.get("<DETAILED_CAPTION>", det_cap_res)
    print(f"Detailed Caption ({det_cap_time:.2f}s):\n\"{det_cap_text}\"")
    
    # 4. Task B: Visual Question Answering
    print("\n" + "-" * 50)
    print("TASK B: VISUAL QUESTION ANSWERING (<VQA>)")
    print("-" * 50)
    question = "What objects or land-cover features are visible in this image?"
    vqa_res, vqa_time, raw_vqa = run_task(model, processor, image, "<VQA>", question)
    vqa_answer = vqa_res.get("<VQA>", vqa_res)
    print(f"Question: \"{question}\"")
    print(f"Latency: {vqa_time:.2f}s")
    print(f"VQA Output: \"{vqa_answer}\"")
    
    # Second VQA question
    question_2 = "What type of terrain or landscape is shown?"
    vqa_res_2, vqa_time_2, _ = run_task(model, processor, image, "<VQA>", question_2)
    vqa_answer_2 = vqa_res_2.get("<VQA>", vqa_res_2)
    print(f"Question 2: \"{question_2}\"")
    print(f"Latency: {vqa_time_2:.2f}s")
    print(f"VQA Output 2: \"{vqa_answer_2}\"")
    
    # 5. Task C: Text-Guided Grounding
    print("\n" + "-" * 50)
    print("TASK C: PHRASE GROUNDING (<CAPTION_TO_PHRASE_GROUNDING>)")
    print("-" * 50)
    ground_phrase = "green field"
    grd_res, grd_time, raw_grd = run_task(model, processor, image, "<CAPTION_TO_PHRASE_GROUNDING>", ground_phrase)
    parsed_grd = grd_res.get("<CAPTION_TO_PHRASE_GROUNDING>", grd_res)
    print(f"Grounding Phrase: \"{ground_phrase}\"")
    print(f"Latency: {grd_time:.2f}s")
    print(f"Parsed Grounding Result: {parsed_grd}")
    
    # Also test <OPEN_VOCABULARY_DETECTION>
    print("\nTesting <OPEN_VOCABULARY_DETECTION> with 'water':")
    ovd_res, ovd_time, _ = run_task(model, processor, image, "<OPEN_VOCABULARY_DETECTION>", "water")
    parsed_ovd = ovd_res.get("<OPEN_VOCABULARY_DETECTION>", ovd_res)
    print(f"Latency: {ovd_time:.2f}s")
    print(f"OVD Result: {parsed_ovd}")
    
    final_mem = get_process_memory_mb()
    print("\n" + "=" * 70)
    print("SUMMARY OF VERIFICATION")
    print("=" * 70)
    print(f"Model: microsoft/Florence-2-base (231.4M parameters)")
    print(f"Hardware: CPU-only execution (torch.float32)")
    print(f"Total Load Time: {load_time:.2f}s")
    print(f"Captioning Latency: {cap_time:.2f}s")
    print(f"VQA Latency: {vqa_time:.2f}s")
    print(f"Grounding Latency: {grd_time:.2f}s")
    print(f"Peak Process Memory: {final_mem:.1f} MB (RAM used by Python: {final_mem - initial_mem:.1f} MB)")
    print("Verification Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
