from flask import Flask, render_template, request
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)

# ----- Modelo difuso (definido una vez al iniciar el servidor) -----
# Universos
temp_univ = np.arange(0, 45.1, 0.5)     # °C
hum_univ  = np.arange(0, 100.1, 1.0)    # %
rain_univ = np.arange(0, 60.5, 0.5)     # mm/semana
freq_univ = np.arange(0, 7.01, 0.05)    # riegos/semana

temp = ctrl.Antecedent(temp_univ, 'temp')
hum  = ctrl.Antecedent(hum_univ,  'hum')
rain = ctrl.Antecedent(rain_univ, 'rain')
freq = ctrl.Consequent(freq_univ, 'freq')

# Membresías
temp['baja']  = fuzz.trapmf(temp.universe, [0, 0, 15, 22])
temp['media'] = fuzz.trimf(temp.universe,  [18, 25, 32])
temp['alta']  = fuzz.trapmf(temp.universe, [28, 38, 45, 45])

hum['baja']   = fuzz.trapmf(hum.universe,  [0, 0, 30, 45])
hum['media']  = fuzz.trimf(hum.universe,   [40, 55, 70])
hum['alta']   = fuzz.trapmf(hum.universe,  [65, 85, 100, 100])

rain['nula']      = fuzz.trapmf(rain.universe, [0, 0, 2, 5])
rain['moderada']  = fuzz.trimf(rain.universe,  [3, 12, 22])
rain['alta']      = fuzz.trapmf(rain.universe, [18, 35, 60, 60])

freq['ninguno'] = fuzz.trapmf(freq.universe, [0, 0, 0.25, 0.75])
freq['bajo']    = fuzz.trimf(freq.universe,  [0.5, 1.5, 2.5])
freq['medio']   = fuzz.trimf(freq.universe,  [2.0, 3.5, 5.0])
freq['alto']    = fuzz.trapmf(freq.universe, [4.5, 6.0, 7.0, 7.0])

# Reglas (las que ya usabas)
rules = [
    ctrl.Rule(rain['alta'],                               freq['ninguno']),
    ctrl.Rule(rain['moderada'] & hum['alta'],             freq['bajo']),
    ctrl.Rule(rain['nula'] & temp['alta'] & hum['baja'],  freq['alto']),

    ctrl.Rule(rain['nula'] & temp['media'] & hum['media'], freq['medio']),
    ctrl.Rule(rain['nula'] & temp['baja']  & hum['alta'],  freq['bajo']),
    ctrl.Rule(rain['moderada'] & temp['alta'] & hum['baja'], freq['medio']),

    ctrl.Rule(rain['nula'] & hum['baja'],                 freq['alto']),
    ctrl.Rule(temp['baja'] & rain['moderada'],            freq['bajo']),
    ctrl.Rule(temp['media'] & rain['moderada'],           freq['medio']),
]

riego_ctrl = ctrl.ControlSystem(rules)

def fuzzy_predict(t, h, r):
    # Seguridad: clamp dentro del universo
    t = float(np.clip(t, temp_univ.min(), temp_univ.max()))
    h = float(np.clip(h, hum_univ.min(),  hum_univ.max()))
    r = float(np.clip(r, rain_univ.min(), rain_univ.max()))

    sim = ctrl.ControlSystemSimulation(riego_ctrl)
    sim.input['temp'] = t
    sim.input['hum']  = h
    sim.input['rain'] = r
    sim.compute()
    val = float(sim.output['freq'])

    # Etiqueta
    if val <= 0.25:
        badge = "Ninguno"
    elif val <= 1.5:
        badge = "Bajo"
    elif val <= 3.5:
        badge = "Medio"
    else:
        badge = "Alto"

    # Texto cada cuántos días
    if val <= 0.25:
        txt = "No se recomienda regar esta semana."
    else:
        dias = 7.0 / max(val, 1e-6)
        txt  = f"≈ cada {dias:.1f} días."

    return val, badge, txt

# ----- Rutas Flask -----
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            t = float(request.form.get("temp", "0"))
            h = float(request.form.get("hum", "0"))
            r = float(request.form.get("rain", "0"))
            val, badge, txt = fuzzy_predict(t, h, r)
            result = {
                "temp": t, "hum": h, "rain": r,
                "freq": round(val, 2),
                "badge": badge,
                "texto": txt
            }
        except Exception as e:
            result = {"error": str(e)}
    return render_template("index.html", result=result)

if __name__ == "__main__":
    # Para desarrollo local
    app.run(debug=True)
