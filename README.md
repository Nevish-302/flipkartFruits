# Flipkart Grid 5.0 - Fruit Shelf Life Backend

## Overview

Welcome to the **Fruit Shelf Life Backend** repository! This Flask application serves as the processing engine for the Fruit Shelf Life Detection system, designed to work seamlessly with the Flipkart Grid 5.0 application. It processes product data extracted from images and predicts the shelf life of fruits using advanced machine learning models.

## Key Features

- **Data Processing**: Efficiently processes raw data received from the Flutter app, ensuring accurate extraction and organization of product details such as names, expiry dates, and prices.
- **Machine Learning Integration**: Utilizes a custom **LLama3-70b model** deployed on a Groq server for real-time data analysis and freshness prediction.
- **Freshness Index Calculation**: Analyzes fruit attributes like shape, color, and texture to accurately predict shelf life.
- **RESTful API**: Exposes endpoints for communication with the Flutter app.

## Tech Stack

- **Flask**: A lightweight WSGI web application framework for Python.
- **Groq**: For deploying and managing machine learning models.
- **AWS**: For cloud deployment and data management.
- **Pandas**: For data manipulation and analysis.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/flask-fruit-shelf-life.git
   cd flask-fruit-shelf-life
   ```
2. **Set up a virtual environment (optional but recommended)**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```
4. **Run the Flask application**:

```bash
flask --app run app.py
```

## Model Architecture

![WhatsApp Image 2024-10-18 at 21 36 40](https://github.com/user-attachments/assets/52c92560-775b-4d6f-abbb-2ce01663a1ce)

## Contributing
Feel free to contribute by submitting issues or pull requests!

## License
This project is licensed under the MIT License.
