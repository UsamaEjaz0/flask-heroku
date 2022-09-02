from flask import Flask, jsonify
# from transformers import DistilBertTokenizerFast
# from transformers import TFDistilBertForSequenceClassification

from transformers import pipeline
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
classifier = pipeline("text-classification", model='philschmid/tiny-bert-sst2-distilled', return_all_scores=True)


# newmodel = TFDistilBertForSequenceClassification.from_pretrained('updated_model',  num_labels=3)
# new_tokenizer = DistilBertTokenizerFast.from_pretrained('tokenizer')

@app.route('/out/<task>', methods=['GET'])
def analyze_task(task):
    return jsonify({'Test': task})


@app.route('/', methods=['GET'])
def main():
    prediction = classifier("Climate change is real", )
    return jsonify({'Test': str(prediction)})


if __name__ == "__main__":
    app.run()


