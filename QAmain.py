from nesy_core.datasets.ambig_qa import AmbigQADataset
from nesy_core.pipeline import PipelineQA  # or AmbigQAPipeline once you build it


# load dataset
dataset = AmbigQADataset.from_hf("train", max_items=100)

# run pipeline
pipeline = PipelineQA()
results = []

for item in dataset:
    prediction = pipeline.run(item)
    results.append({
        "id":         item["id"],
        "question":   item["question"],
        "prediction": prediction,
        "answer":     item["answer"],
    })

# evaluate
predictions   = [r["prediction"] for r in results]
ground_truths = [r["answer"] for r in results]

metrics = AmbigQADataset.evaluate(predictions, ground_truths)
print(metrics)