import app
import numpy as np
print("Testing predict_digit...")
img = np.zeros((28, 28))
try:
    res = app.predict_digit(img)
    print("predict_digit passed!", res)
except Exception as e:
    print("predict_digit failed:", e)

print("Testing generate_text...")
try:
    text, fig = app.generate_text("O Romeo,", max_tokens=10, temperature=0.8)
    print("generate_text passed!", text)
except Exception as e:
    print("generate_text failed:", e)
