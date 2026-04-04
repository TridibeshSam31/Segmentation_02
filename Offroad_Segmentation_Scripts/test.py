import torch
import numpy as np
from transformers import Mask2FormerForUniversalSegmentation

def benchmark_inference_speed(model_dir, image_size=(518, 518), iterations=100):
    # 1. Setup device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    print("Loading model...")
    # This automatically reads your config.json and model.safetensors
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir).to(device)
    model.eval()

    # 2. Create a dummy input tensor
    # We use a random tensor instead of a real image to test PURE GPU compute speed.
    # This removes disk I/O (reading from a hard drive) from the speed test.
    dummy_input = torch.randn(1, 3, image_size[0], image_size[1], dtype=torch.float32).to(device)

    # 3. Warm-up Phase
    # The first few inference passes on a GPU are always much slower due to 
    # memory allocation. We run 10 dummy passes to "spin up" the GPU.
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(pixel_values=dummy_input)

    # 4. Benchmarking Phase
    print(f"Running {iterations} iterations for benchmarking...")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((iterations, 1))

    with torch.no_grad():
        for i in range(iterations):
            starter.record()
            
            # Forward pass
            _ = model(pixel_values=dummy_input)
            
            ender.record()
            # This forces Python to wait until the GPU physically finishes the math
            torch.cuda.synchronize() 
            
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time

    # 5. Calculate Metrics
    mean_syn = np.sum(timings) / iterations
    std_syn = np.std(timings)
    fps = 1000 / mean_syn

    # 6. Print Results
    print("\n" + "="*50)
    print("🚀 INFERENCE SPEED RESULTS")
    print("="*50)
    print(f"Model Directory:  {model_dir}")
    print(f"Input Resolution: {image_size[0]}x{image_size[1]}")
    print(f"Average Latency:  {mean_syn:.2f} milliseconds per frame")
    print(f"Latency Std Dev:  {std_syn:.2f} ms")
    print(f"Estimated FPS:    {fps:.2f} Frames Per Second")
    print("="*50)

if __name__ == "__main__":
    # Point this to the exact folder containing your model.safetensors
    # Since your screenshot shows you are inside the folder, "." means current directory.
    MODEL_DIRECTORY = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Scripts\best_offroad_model_refined"  
    
    benchmark_inference_speed(MODEL_DIRECTORY, image_size=(518, 518))