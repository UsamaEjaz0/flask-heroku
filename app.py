from flask import Flask, jsonify
# from transformers import DistilBertTokenizerFast
# from transformers import TFDistilBertForSequenceClassification

from transformers import pipeline
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
# classifier = pipeline("text-classification",model='distilbert-base-uncased-finetuned-sst-2-english', return_all_scores=True)
qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-small-finetuned-squadv2",
    tokenizer="mrm8488/bert-small-finetuned-squadv2"
)


# newmodel = TFDistilBertForSequenceClassification.from_pretrained('updated_model',  num_labels=3)
# new_tokenizer = DistilBertTokenizerFast.from_pretrained('tokenizer')

@app.route('/out/<task>', methods=['GET'])
def analyze_task(task):
    return jsonify({'Test': task})


@app.route('/', methods=['GET'])
def main():
    print("Inside main")
    # predict_input = new_tokenizer.encode("climate change is man made concept", return_tensors="tf")
    # prediction = classifier("Climate change is real", )
    prediction = qa_pipeline({
        'context': "Manuel Romero has been working hardly in the repository hugginface/transformers lately",
        'question': "Who has been working hard for hugginface/transformers lately?"

    })
    return jsonify({'Test': str(prediction)})


if __name__ == "__main__":
    app.run()


