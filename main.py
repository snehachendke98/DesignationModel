from flask import Flask , request,jsonify
# from helpers import helper
import logging
from Model import designationModel

app = Flask(__name__)
file_handler = logging.FileHandler('AIServices.log')
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.ERROR)

@app.route('/sneha-AI/heartbeat',methods=['GET','POST'])
def heartbeat():
    return jsonify({"response" : True})

@app.route('/sneha-AI/designation_service',methods=['GET','POST'])
def designation_service():
    try:
        serviceInput = request.json['serviceInput']
        serviceName = request.json['serviceName']
        if serviceName == 'DESIGNATION':
            return designationModel.classification(serviceInput),{'Content-Type':'application/json'}
    except Exception as error:
        app.logger.exception(error)
        return jsonify({'errorMessage': str(error)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5003)