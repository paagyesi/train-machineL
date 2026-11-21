import joblib
import pandas as pd

model = joblib.load("model.pkl")

new_data = pd.DataFrame(
    {
        "age": [39],
        "salary": [70000]
    }
)
predictions = model.predict(new_data)

if predictions[0] == 1:
    print("The customer is likely to buy the product.")

else:
    print("The customer is unlikely to buy the product.")

print(f"Predicted class: {predictions[0]}")