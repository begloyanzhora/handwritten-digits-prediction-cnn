# Handwritten Digit Recognition using Convolutional Neural Network

## How to Run Project
(Run all commands within app.py directory)
1. Install Python virtual environment:
    ```
    $ python3 -m venv <venv_name>
    ```
2. Create and activate virtual environment:
    ```
    $ python3 -m venv <venv_name>
    $ source <venv_name>/bin/activate
    ```
3. Install all requirements:
    ```
    $ pip install -r requirements.txt
    ```
4. Run server:
    ```
    python app.py
    ```
5. Open in browser: http://localhost:5000

## Prediction Accuracies
- On test data
    - Loss = 3.539
    - Accuracy = 0.99
- Best prediction - 0
    - predicted with 100% accuracy (972 from 972)
- Worst prediction - 6
    - predicted with 98% accuracy (942 from 956)
