

import re
import easyocr
import os

def judge(result):

    # Retrieve the recognized text
    texts = [text for (_, text, _) in result]
    new_test_list = []
    for t in texts:
        # re.findall  ['Grasshoppers']
        letters = re.findall(r'[A-Za-z]+', t)
        print(letters)
        new_test_list.extend(letters)

    # print("Recognized texts:", texts)

    # If there are no words or only one word, it's incomparable.
    if len(new_test_list) <= 1:
        print("Not enough text to compare.")
        return False
    else:
        # Use all lowercase letters and remove spaces to avoid misjudgments caused by case sensitivity.
        normalized = [t.lower().strip() for t in new_test_list]

        # Check if all are the same
        all_same = all(t == normalized[0] for t in normalized)

        print("\n=== Comparison Result ===")
        if all_same:
            print("All recognized texts are the SAME.")
            print(f"Unified text: {normalized[0]}")
            return True
        else:
            print("Recognized texts are DIFFERENT.")
            for i, t in enumerate(normalized, start=1):
                print(f"Text {i}: {t}")
            return False

if __name__ == '__main__':
    reader = easyocr.Reader(['en'])
    # root = r'C:\Users\shenyanjian\Downloads\mmdetection-dataclened\eval\output2_show (1)\output2_show'
    root = r'C:\Users\shenyanjian\Downloads\mmdetection-non-dataclened\eval\output2_show (1)\output2_show'
    files = os.listdir(root)
    true_list = 0
    false_list = 0
    for f in files:
        result = reader.readtext(os.path.join(root, f))
        judge_correct_bool = judge(result)
        print(judge_correct_bool)
        if judge_correct_bool==True:
            true_list += 1
        else:
            false_list += 1
    print("Totle file num", len(files))
    print("true_list", true_list)
    print("false_list", false_list)

