success_cases = [
    r for r in results["reflexion"]
    if not r["first_em"] and r["em"]
][:3]

for i, case in enumerate(success_cases, 1):
    print(f"===== Case {i} =====")
    print(f"Question: {case['question']}")
    print(f"Gold: {case['gold']}")
    print(f"First EM: {case['first_em']} | First F1: {case['first_f1']:.2f}")
    print(f"First (wrong): {case['first_pred']}")
    print(f"Reflection: {case['reflection']}")
    print(f"Second (correct): {case['second_pred']}")
    print(f"Tokens used: {case['tokens']} (first {case['first_tokens']}, reflect {case['reflect_tokens']}, second {case['second_tokens']})\n")

if not success_cases:
    print("No successful Reflexion cases found in this sample set.")
