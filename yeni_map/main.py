from flask import Flask, render_template, request, jsonify
import pandas as pd
import unicodedata

app = Flask(__name__)

# CSV dosyasını yükle
df = pd.read_csv("new_optimum_yesil_alan_sonuclari_SCENARIOS.csv")

# İlçe adlarını normalize eden fonksiyon
def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    tr_map = str.maketrans("çğıöşü", "cgiosu")
    name = name.translate(tr_map)
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    return name.replace(" ", "")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    w1 = float(request.args.get("w1", 0.5))
    w2 = float(request.args.get("w2", 0.5))
    algo = request.args.get("algo", "GA")

    col_name = f"Yeni_Yapilacak_Yesil_Alan_{algo}_W1_{w1:.1f}_W2_{w2:.1f}"
    if col_name not in df.columns:
        return jsonify({})

    data = {
        normalize_name(row["ILCE"]): row[col_name]
        for _, row in df.iterrows()
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
