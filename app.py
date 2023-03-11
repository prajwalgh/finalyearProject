

from flask import Flask,request, render_template

app=Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    #return "first page"
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        file_path = 'uploads'
        f.save(file_path + '/' + f.filename)
        # Make prediction
        preds = model_predict(file_path + '/' + f.filename, model_loaded)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)  # ImageNet Decode
        result = str(pred_class[0][0][1])  # Convert to string
        return result
    return None
if __name__ == "__main__":
    app.run(debug=True)
