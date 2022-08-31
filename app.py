from flask import Flask, jsonify
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


app = Flask(__name__)


newmodel = TFDistilBertForSequenceClassification.from_pretrained('updated_model',  num_labels=3)
new_tokenizer = DistilBertTokenizerFast.from_pretrained('tokenizer')


@app.route('/out/<task>', methods=['GET'])
def analyze_task(task):
    return jsonify({'Test': task})


@app.route('/', methods=['GET'])
def main():
    print("Inside main")
    predict_input = new_tokenizer.encode("climate change is man made concept", return_tensors="tf")
    return jsonify({'Test': "Fuck you"})


if __name__ == "__main__":
    app.run()
