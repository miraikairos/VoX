const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const cors = require("cors");

const app = express();

app.use(cors());

const upload = multer({ dest: "uploads/" });

app.post("/analyze", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: "No audio uploaded"
      });
    }

    const form = new FormData();

    form.append(
      "file",
      fs.createReadStream(req.file.path),
      req.file.originalname
    );

    // Send audio to Python AI
    const response = await axios.post(
      "http://127.0.0.1:8000/predict",
      form,
      {
        headers: form.getHeaders()
      }
    );

    // Delete temporary file
    fs.unlinkSync(req.file.path);

    res.json(response.data);

  } catch (error) {

    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});


app.listen(5000, () => {
  console.log("Backend running on http://localhost:5000");
});