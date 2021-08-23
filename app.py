from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {0 : 'Infected', 1 : 'Uninfected'}

model = load_model('MaleriaBineryClassifier_A_971_VA_959.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(64,64))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 64,64,3)
	maleria_predict_output = model.predict_classes(i)
	maleria_prict_prob = model.predict_proba(i)
	return dic[maleria_predict_output[0]],maleria_prict_prob ;


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "UESC College Project"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/cell_image/" + img.filename	
		img.save(img_path)
		maleria_predict_output,maleria_prict_prob = predict_label(img_path)
	return render_template("index.html", prediction = maleria_predict_output, infected = maleria_prict_prob[0][0],uninfected = maleria_prict_prob[0][1], img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)