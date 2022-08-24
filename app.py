from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/monday/<task>', methods=['GET'])
def analyze_task(task):

    return jsonify({'Test': task})


@app.route('/', methods=['GET'])
def main():
    return jsonify({'Test': "Working"})


if __name__ == "__main__":
    app.run()
