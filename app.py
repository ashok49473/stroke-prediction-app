from util import *
####################### Flask settings #######################
app = Flask(__name__)

####################### Home Page ############################
@app.route("/", methods=['GET'])
@app.route("/home", methods=['GET'])
def home():
   return render_template("index.html")

########################## Results Page #######################
@app.route("/result", methods = ["GET","POST"])
def predict():
	if request.method == 'POST':
		feature_dict = request.form
		X = []
		for col in features:
			if col in ['hypertension', 'heart_disease']:
				X.append(int(feature_dict[col]))
			else:
				X.append(feature_dict[col])
		proba = round(predict_class([X]),2)
		return render_template("index.html", proba=proba)
####################################################

@app.route("/about")
def about():
	return render_template("about.html")

if __name__ == '__main__':

	app.run(debug=True)
