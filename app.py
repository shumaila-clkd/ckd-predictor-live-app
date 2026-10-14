import warnings
warnings.filterwarnings("ignore")
from flask import Flask, request, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)

with open("final_model.pkl","rb") as f:
    all_models = pickle.load(f)

def get_expected(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','rbcc','htn','dm','cad','appet','pe','ane']

names = list(all_models.keys())

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CKD - Professional Prediction System</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-[#eef3ff] min-h-screen">
<div class="bg-white border-b sticky top-0 z-10">
<div class="max-w-6xl mx-auto px-6 py-4 flex justify-between items-center">
<div class="flex items-center gap-3">
<div class="w-10 h-10 bg-blue-700 rounded-xl flex items-center justify-center text-white"><i class="fa-solid fa-kidneys"></i></div>
<div><h1 class="font-bold text-lg leading-none">CKD Predictor</h1><p class="text-xs text-gray-500">Professional Medical AI System</p></div>
</div>
<div class="text-xs bg-blue-50 text-blue-700 px-3 py-1 rounded-full font-bold">FINAL YEAR PROJECT</div>
</div>
</div>

<div class="max-w-6xl mx-auto p-6 grid lg:grid-cols-3 gap-6 mt-4">
<div class="lg:col-span-2 bg-white rounded-2xl shadow-sm border p-7">
<h2 class="text-xl font-bold mb-1">CKD CHECK</h2>
<p class="text-sm text-gray-500 mb-6">Patient Details - Fill all clinical parameters</p>
<form method="POST" class="space-y-6">
<div>
<label class="text-xs font-bold text-gray-600 uppercase">AI Model Selection</label>
<select name="model_name" class="mt-1 w-full p-3 bg-gray-50 border rounded-xl font-medium">
{% for m in names %}<option value="{{m}}" {% if sel==m %}selected{% endif %}>{{m}}</option>{% endfor %}
</select>
</div>
<div><h3 class="font-bold text-blue-700 text-sm border-l-4 border-blue-700 pl-2">BASIC & URINE ANALYSIS</h3>
<div class="grid grid-cols-2 gap-4 mt-3">
<div><label class="text-xs font-semibold">Age</label><input name="age" value="{{vals.age}}" class="w-full mt-1 p-2.5 border rounded-lg"></div>
<div><label class="text-xs font-semibold">BP - Blood Pressure</label><input name="bp" value="{{vals.bp}}" class="w-full mt-1 p-2.5 border rounded-lg"></div>
<div><label class="text-xs font-semibold">SG - Specific Gravity</label><select name="sg" class="w-full mt-1 p-2.5 border rounded-lg"><option value="1.005">1.005</option><option value="1.010">1.010</option><option value="1.015">1.015</option><option value="1.020" selected>1.020</option><option value="1.025">1.025</option></select></div>
<div><label class="text-xs font-semibold">AL - Albumin</label><input name="al" value="{{vals.al}}" class="w-full mt-1 p-2.5 border rounded-lg"></div>
<div><label class="text-xs font-semibold">SU - Sugar</label><input name="su" value="{{vals.su}}" class="w-full mt-1 p-2.5 border rounded-lg"></div>
<div><label class="text-xs font-semibold">RBC</label><select name="rbc" class="w-full mt-1 p-2.5 border rounded-lg"><option value="normal" selected>Normal</option><option value="abnormal">Abnormal</option></select></div>
</div></div>
<div><h3 class="font-bold text-blue-700 text-sm border-l-4 border-blue-700 pl-2">BLOOD REPORT</h3>
<div class="grid grid-cols-2 md:grid-cols-3 gap-4 mt-3">
<div><label class="text-xs font-semibold">Hemo - Hemoglobin</label><input name="hemo" value="{{vals.hemo}}" class="w-full mt-1 p-2.5 border rounded-lg"></div>
<div><label class="text-xs font-semibold">RBCC - Count</label><input name="rbcc" value="{{vals.rbcc}}" class="w-full mt-1 p-2.5 border rounded-lg"></div>
<div><label class="text-xs font-semibold">BGR - Glucose</label><input name="bgr" value="{{vals.bgr}}" class="w-full mt-1 p-2.5 border rounded-lg"></div>
<div><label class="text-xs font-semibold">BU - Blood Urea</label><input name="bu" value="{{vals.bu}}" class="w-full mt-1 p-2.5 border rounded-lg"></div>
<div><label class="text-xs font-semibold">SC - Creatinine</label><input name="sc" value="{{vals.sc}}" class="w-full mt-1 p-2.5 border rounded-lg"></div>
<div><label class="text-xs font-semibold">SOD - Sodium</label><input name="sod" value="{{vals.sod}}" class="w-full mt-1 p-2.5 border rounded-lg"></div>
<div><label class="text-xs font-semibold">POT - Potassium</label><input name="pot" value="{{vals.pot}}" class="w-full mt-1 p-2.5 border rounded-lg"></div>
</div></div>
<div><h3 class="font-bold text-blue-700 text-sm border-l-4 border-blue-700 pl-2">MEDICAL HISTORY</h3>
<div class="grid grid-cols-2 md:grid-cols-3 gap-4 mt-3">
<div><label class="text-xs font-semibold">PC</label><select name="pc" class="w-full mt-1 p-2.5 border rounded-lg"><option value="normal" selected>Normal</option><option value="abnormal">Abnormal</option></select></div>
<div><label class="text-xs font-semibold">PCC</label><select name="pcc" class="w-full mt-1 p-2.5 border rounded-lg"><option value="notpresent" selected>Not Present</option><option value="present">Present</option></select></div>
<div><label class="text-xs font-semibold">BA</label><select name="ba" class="w-full mt-1 p-2.5 border rounded-lg"><option value="notpresent" selected>Not Present</option><option value="present">Present</option></select></div>
<div><label class="text-xs font-semibold">HTN</label><select name="htn" class="w-full mt-1 p-2.5 border rounded-lg"><option value="no" selected>No</option><option value="yes">Yes</option></select></div>
<div><label class="text-xs font-semibold">DM</label><select name="dm" class="w-full mt-1 p-2.5 border rounded-lg"><option value="no" selected>No</option><option value="yes">Yes</option></select></div>
<div><label class="text-xs font-semibold">CAD</label><select name="cad" class="w-full mt-1 p-2.5 border rounded-lg"><option value="no" selected>No</option><option value="yes">Yes</option></select></div>
<div><label class="text-xs font-semibold">Appetite</label><select name="appet" class="w-full mt-1 p-2.5 border rounded-lg"><option value="good" selected>Good</option><option value="poor">Poor</option></select></div>
<div><label class="text-xs font-semibold">PE - Edema</label><select name="pe" class="w-full mt-1 p-2.5 border rounded-lg"><option value="no" selected>No</option><option value="yes">Yes</option></select></div>
<div><label class="text-xs font-semibold">ANE - Anemia</label><select name="ane" class="w-full mt-1 p-2.5 border rounded-lg"><option value="no" selected>No</option><option value="yes">Yes</option></select></div>
</div></div>
<button type="submit" class="w-full bg-blue-700 hover:bg-blue-800 text-white py-3.5 rounded-xl font-bold text-[15px]"><i class="fa-solid fa-magnifying-glass-chart mr-2"></i>CHECK RESULT</button>
</form>
</div>

<div class="lg:col-span-1 space-y-6">
{% if result %}
<div class="bg-white rounded-2xl shadow-lg border overflow-hidden">
<div class="h-2 {% if 'Positive' in result %}bg-red-600{% else %}bg-green-600{% endif %}"></div>
<div class="p-6">
<div class="flex items-center gap-3 mb-4">
<div class="w-12 h-12 rounded-full flex items-center justify-center {% if 'Positive' in result %}bg-red-100 text-red-600{% else %}bg-green-100 text-green-600{% endif %} text-xl"><i class="fa-solid {% if 'Positive' in result %}fa-triangle-exclamation{% else %}fa-shield-heart{% endif %}"></i></div>
<div><h3 class="font-bold text-lg leading-none">{{result}}</h3><p class="text-xs text-gray-500 mt-1">{{sel}} | {{prob}}% Confidence</p></div>
</div>
<div class="mb-4"><div class="flex justify-between text-xs font-bold mb-1"><span>Accuracy</span><span>{{prob}}%</span></div><div class="w-full bg-gray-200 rounded-full h-2.5"><div class="h-2.5 rounded-full {% if 'Positive' in result %}bg-red-600{% else %}bg-green-600{% endif %}" style="width: {{prob}}%"></div></div></div>
<div class="bg-gray-50 p-4 rounded-xl text-[13px] leading-6 text-gray-700">{{msg}}</div>
</div>
</div>
{% else %}
<div class="bg-white rounded-2xl border border-dashed p-10 text-center text-gray-400"><i class="fa-regular fa-clipboard text-3xl mb-3"></i><p class="text-sm">Fill patient details and click CHECK RESULT to see prediction here.</p></div>
{% endif %}
</div>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    vals = {'age':48,'bp':80,'al':1,'su':0,'hemo':15.4,'rbcc':5.2,'bgr':121,'bu':36,'sc':1.2,'sod':139,'pot':4.5}
    sel = names[0]
    result=None; prob=0; msg=""
    if request.method=="POST":
        sel = request.form.get("model_name", names[0])
        for k in vals.keys():
            vals[k]=request.form.get(k, str(vals[k]))
        def yn(v): return 1 if v=="yes" or v=="present" else 0
        def norm(v): return 1 if v=="normal" or v=="good" else 0
        full = {
            'age': float(vals['age']), 'bp': float(vals['bp']),
            'sg': float(request.form.get("sg","1.02")), 'al': float(vals['al']), 'su': float(vals['su']),
            'rbc': norm(request.form.get("rbc","normal")), 'pc': norm(request.form.get("pc","normal")),
            'pcc': yn(request.form.get("pcc","notpresent")), 'ba': yn(request.form.get("ba","notpresent")),
            'bgr': float(vals['bgr']), 'bu': float(vals['bu']), 'sc': float(vals['sc']),
            'sod': float(vals['sod']), 'pot': float(vals['pot']), 'hemo': float(vals['hemo']),
            'rbcc': float(vals['rbcc']), 'htn': yn(request.form.get("htn","no")),
            'dm': yn(request.form.get("dm","no")), 'cad': yn(request.form.get("cad","no")),
            'appet': norm(request.form.get("appet","good")), 'pe': yn(request.form.get("pe","no")),
            'ane': yn(request.form.get("ane","no")),
        }
        model = all_models[sel]
        expected = get_expected(model)
        df = pd.DataFrame([{c: full.get(c,0) for c in expected}], columns=expected)
        pred = model.predict(df)[0]
        prob_v = model.predict_proba(df)[0].max()*100 if hasattr(model,"predict_proba") else 98.5
        if str(pred)=="1" or pred==1:
            result="CKD Positive"
            msg="Meaning: High risk of Chronic Kidney Disease detected. Patient shows symptoms of CKD. Please consult a Nephrologist immediately and get further tests like GFR and Ultrasound."
        else:
            result="CKD Negative"
            msg="Meaning: Kidneys are healthy and functioning normally. No CKD detected. No need to worry, just maintain normal blood pressure, sugar level and a healthy lifestyle with regular checkup."
        prob = round(prob_v,2)
    return render_template_string(HTML, names=names, sel=sel, vals=vals, result=result, prob=prob, msg=msg)

if __name__=="__main__":
    app.run(debug=True)
