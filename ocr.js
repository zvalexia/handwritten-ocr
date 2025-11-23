var ocrDemo = {
    CANVAS_WIDTH: 200,
    TRANSLATED_WIDTH: 20,
    PIXEL_WIDTH: 10,
    HOST: "http://localhost",
    PORT: "8080",

    data: [],
    trainArray: [],
    trainingRequestCount: 0,
    BATCH_SIZE: 5,

    onLoadFunction: function() {
        this.canvas = document.getElementById("canvas");
        this.ctx = this.canvas.getContext("2d");

        this.resetCanvas();

        this.canvas.onmousedown = (e) => { this.isDrawing = true; this.fillFromMouse(e); }
        this.canvas.onmouseup   = () => { this.isDrawing = false; }
        this.canvas.onmousemove = (e) => { if (this.isDrawing) this.fillFromMouse(e); }
    },

    resetCanvas: function() {
        this.data = Array(400).fill(0);
        this.ctx.fillStyle = "black";
        this.ctx.fillRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH);
        this.drawGrid();
    },

    drawGrid: function() {
        this.ctx.strokeStyle = "#333";
        for (let i = 0; i <= this.CANVAS_WIDTH; i += this.PIXEL_WIDTH) {
            this.ctx.beginPath();
            this.ctx.moveTo(i, 0);
            this.ctx.lineTo(i, this.CANVAS_WIDTH);
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(0, i);
            this.ctx.lineTo(this.CANVAS_WIDTH, i);
            this.ctx.stroke();
        }
    },

    fillFromMouse: function(e) {
        let rect = this.canvas.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;
        this.fillSquare(x, y);
    },

    fillSquare: function(x, y) {
        let xp = Math.floor(x / this.PIXEL_WIDTH);
        let yp = Math.floor(y / this.PIXEL_WIDTH);

        if (xp < 0 || xp >= 20 || yp < 0 || yp >= 20) return;

        // 3x3 stroke thickness
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                let xx = xp + dx;
                let yy = yp + dy;

                if (xx < 0 || xx >= 20 || yy < 0 || yy >= 20) continue;

                let id = yy * 20 + xx;
                this.data[id] = 1;

                this.ctx.fillStyle = "white";
                this.ctx.fillRect(
                    xx * this.PIXEL_WIDTH,
                    yy * this.PIXEL_WIDTH,
                    this.PIXEL_WIDTH,
                    this.PIXEL_WIDTH
                );
            }
        }
    },

    train: function() {
        let d = document.getElementById("digit").value;
        if (d === "" || isNaN(d)) {
            alert("Enter a digit between 0 and 9 to train.");
            return;
        }

        this.trainArray.push({ y0: this.data.slice(), label: parseInt(d) });
        this.trainingRequestCount++;

        if (this.trainingRequestCount >= this.BATCH_SIZE) {
            this.sendData({ train: true, trainArray: this.trainArray });
            this.trainArray = [];
            this.trainingRequestCount = 0;
        }

        this.resetCanvas();
    },

    test: function() {
        this.sendData({ predict: true, image: this.data });
    },

    sendData: function(json) {
        let xhr = new XMLHttpRequest();
        xhr.open("POST", `${this.HOST}:${this.PORT}`, true);
        xhr.setRequestHeader("Content-Type", "application/json");

        xhr.onload = () => {
            try {
                let res = JSON.parse(xhr.responseText);
                if (res.type === "test") {
                    alert("Prediction: " + res.result);
                }
            } catch (e) {
                console.error("Bad response", e);
            }
        };

        xhr.send(JSON.stringify(json));
    },

    stopServer: function() {
        let xhr = new XMLHttpRequest();
        xhr.open("POST", `${this.HOST}:${this.PORT}`, true);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.onload = () => {
            alert("Server stopped.");
        };
        xhr.send(JSON.stringify({ stop: true }));
    }
};
