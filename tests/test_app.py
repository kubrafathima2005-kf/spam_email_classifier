import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from app import app


def test_health():
    response = app.test_client().get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'ok'


def test_spam_prediction():
    response = app.test_client().post('/predict', json={
        'subject': 'You won a free prize',
        'body': 'Claim your cash reward now by clicking this link.'
    })
    assert response.status_code == 200
    assert response.json['is_spam'] is True


def test_legitimate_prediction():
    response = app.test_client().post('/predict', json={
        'subject': 'Project meeting tomorrow',
        'body': 'Please bring your progress notes to the team meeting.'
    })
    assert response.status_code == 200
    assert response.json['is_spam'] is False
