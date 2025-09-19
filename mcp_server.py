from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/.well-known/mcp.json", methods=["GET"])
def discovery():
    return jsonify({
        "tools": [
            {
                "name": "get_patient_data",
                "description": "Fetch patient details by ID",
                "parameters": {"patient_id": "string"}
            },
            {
                "name": "run_diagnostic_test",
                "description": "Run a diagnostic test for a patient",
                "parameters": {"patient_id": "string"}
            }
        ]
    })

@app.route("/invoke/get_patient_data", methods=["POST"])
def get_patient_data():
    data = request.json
    patient_id = data.get("patient_id", "")
    # Simulated patient data
    return jsonify({
        "patient_id": patient_id,
        "name": "Patient X",
        "age": 45,
        "history": "Hypertension"
    })

@app.route("/invoke/run_diagnostic_test", methods=["POST"])
def run_diagnostic_test():
    data = request.json
    patient_id = data.get("patient_id", "")
    # Simulated test result
    return jsonify({
        "patient_id": patient_id,
        "status": "Test completed",
        "result": "Normal"
    })

if __name__ == "__main__":
    app.run(port=5000)
