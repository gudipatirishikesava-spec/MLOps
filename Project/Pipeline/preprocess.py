import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\MLOps\Project\Dataset\student-por.csv", sep=";")

# Save processed dataset
df.to_csv("processed_student_data.csv", index=False)

print("Data Preprocessing Completed")