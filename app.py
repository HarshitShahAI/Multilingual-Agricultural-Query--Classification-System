from flask import Flask, request, render_template, redirect, url_for
from modelrouter import ModelHandler  # Assuming ModelHandler is your classifier handler

app = Flask(__name__, template_folder="templates", static_folder="static")

# Dictionaries to map label numbers to descriptions
label_description_english = {
    0: "Cultural Practices",
    1: "Fertilizer Use and Availability",
    2: "Field Preparation",
    3: "Government Schemes",
    4: "Market Information",
    5: "Nutrient Management",
    6: "Plant Protection",
    7: "Varieties",
    8: "Weather",
    9: "Weed Management"
}

label_description_gujarati = {
    0: "સાંસ્કૃતિક પ્રથાઓ",
    1: "ખાતરનો ઉપયોગ અને ઉપલબ્ધતા",
    2: "ક્ષેત્રની તૈયારી",
    3: "સરકારી યોજનાઓ",
    4: "બજાર માહિતી",
    5: "પોષક તત્ત્વોનું સંચાલન",
    6: "છોડનું રક્ષણ",
    7: "જાતો",
    8: "હવામાન",
    9: "નીંદણ વ્યવસ્થાપન"
}

label_description_hindi = {
    0: "सांस्कृतिक प्रथाएँ",
    1: "उर्वरक का उपयोग और उपलब्धता",
    2: "क्षेत्र की तैयारी",
    3: "सरकारी योजनाएं",
    4: "बाजार जानकारी",
    5: "पोषक तत्व प्रबंधन",
    6: "पौधे की सुरक्षा",
    7: "किस्में",
    8: "मौसम",
    9: "खरपतवार प्रबंधन"
}

# Helper function to map class number -> readable name
def get_label_description(label, lang="guj"):
    if lang == "eng":
        lookup = label_description_english
    elif lang == "guj":
        lookup = label_description_gujarati
    elif lang == "hin":
        lookup = label_description_hindi
    else:
        raise ValueError("Unsupported language code. Use 'eng', 'guj', or 'hin'.")
    
    return lookup.get(label, "Unknown label")

@app.route('/')
def home():
    # Redirect from the home route (127.0.0.1:5000) to /form
    return redirect(url_for('form'))

@app.route('/form', methods=['GET', 'POST'])
def form():
    result = {}
    if request.method == 'POST':
        query = request.form['query']
        lang = request.form['lang']

        handler = ModelHandler(lang)
        raw_pred = handler.predict(query)
        predicted_label = int(raw_pred)    # cast here too
        print(f"Predicted label = {predicted_label}")  # Print predicted label
        description = get_label_description(predicted_label, lang)

        result = {
            "query": query,
            "language": lang,
            "label": predicted_label,
            "description": description
        }

    return render_template("form.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
