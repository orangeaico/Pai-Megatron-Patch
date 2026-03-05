import json
import numpy as np

def softmax(logits, temperature=1.0):
    """Calculate softmax with temperature scaling"""
    # Divide by temperature
    scaled_logits = np.array(logits) / temperature
    # Subtract max for numerical stability
    scaled_logits = scaled_logits - np.max(scaled_logits)
    # Calculate softmax
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits)

# Load the JSON file
with open('/home/shared/megatron_dir/data/distillation_data/django__django-14238_logprobs.json', 'r') as f:
    js = json.load(f)

# Process first 5 positions
for i in range(5):
    logits = js["teacher_logits"]["values"][i]
    
    # Calculate softmax with T=1 and T=3
    softmax_t1 = softmax(logits, temperature=1.0)
    softmax_t3 = softmax(logits, temperature=2.0)
    
    print(f"\nPosition {i}:")
    print(f"Number of logits: {len(logits)}")
    print("\nFirst 10 values comparison:")
    print("Index | Softmax (T=1)    | Softmax (T=3)")
    print("-" * 50)
    
    # Show first 10 values for comparison
    for j in range(min(10, len(logits))):
        print(f"{j:5d} | {softmax_t1[j]:14.8f} | {softmax_t3[j]:14.8f}")
    
    # Show statistics
    print(f"\nMax probability (T=1): {np.max(softmax_t1):.8f}")
    print(f"Max probability (T=3): {np.max(softmax_t3):.8f}")
    print(f"Entropy (T=1): {-np.sum(softmax_t1 * np.log(softmax_t1 + 1e-10)):.8f}")
    print(f"Entropy (T=3): {-np.sum(softmax_t3 * np.log(softmax_t3 + 1e-10)):.8f}")