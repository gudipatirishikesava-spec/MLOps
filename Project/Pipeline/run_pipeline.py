import subprocess

print("Starting ML Pipeline...\n")

subprocess.run(["python", "D:\MLOps\Project\Pipeline\preprocess.py"])

subprocess.run(["python", "D:\MLOps\Project\Pipeline\Train.py"])

subprocess.run(["python", "D:\MLOps\Project\Pipeline\predict.py"])

print("\nPipeline Executed Successfully")