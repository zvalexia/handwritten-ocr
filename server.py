import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from ocr import OCRNeuralNetwork
import os

nn = OCRNeuralNetwork(hidden_nodes=15)

# Persistent training dataset
TRAIN_DATA_FILE = "train_data.json"
if os.path.exists(TRAIN_DATA_FILE):
    with open(TRAIN_DATA_FILE) as f:
        train_data = json.load(f)
else:
    train_data = []

class OCRHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))

        if "stop" in payload:
            response = {"status": "stopping"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
            # Shutdown server after responding
            def shutdown_server():
                self.server.shutdown()
            import threading
            threading.Thread(target=shutdown_server).start()
            return

        elif "train" in payload:
            # Add new samples to persistent training data
            train_data.extend(payload["trainArray"])
            with open(TRAIN_DATA_FILE, "w") as f:
                json.dump(train_data, f)
            # Train on all accumulated data
            nn.train_batch(train_data)
            nn.save()
            response = {"status": "ok"}

        elif "predict" in payload:
            print("Prediction input sum:", sum(payload["image"]))
            result = nn.predict(payload["image"])
            response = {"type": "test", "result": result}

        else:
            self.send_error(400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run():
    server = HTTPServer(("0.0.0.0", 8080), OCRHandler)
    print("Server running on port 8080")
    server.serve_forever()

if __name__ == "__main__":
    run()
