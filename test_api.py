from io import BytesIO

import pytest
from backend.app import app

@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client

def make_session(client):
    response = client.post('/api/session')
    assert response.status_code == 200
    return response.get_json()['session_id']

def sample_rows():
    rows = []
    for i in range(60):
        units = (i % 7) + 1
        price = 10 + (i % 5)
        rows.append({'OrderID': i + 1, 'Region': ['North','South','East'][i % 3],
                     'Product': ['Laptop','Phone','Tablet'][i % 3],
                     'Units': units, 'UnitPrice': price,
                     'Sales': units * price, 'Profit': units * price * 0.2})
    return rows

def upload_sample(client, sid, rows=None):
    rows = rows or sample_rows()
    return client.post('/api/dataset', headers={'X-Session-Id': sid}, json={
        'name':'test.csv', 'columns':list(rows[0].keys()), 'rows':rows})

def test_session_required_and_isolated(client):
    assert client.post('/api/dataset', json={'rows':[{'x':1}], 'columns':['x']}).status_code == 401
    sid1, sid2 = make_session(client), make_session(client)
    assert upload_sample(client, sid1).status_code == 200
    response = client.get('/api/profile', headers={'X-Session-Id':sid2})
    assert response.status_code == 400

def test_profile_chat_insights_and_read_only_sql(client, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    sid = make_session(client); assert upload_sample(client, sid).status_code == 200
    headers={'X-Session-Id':sid}
    profile=client.get('/api/profile',headers=headers)
    assert profile.status_code == 200 and profile.get_json()['profile']['rows'] == 60
    chat=client.post('/api/chat',headers=headers,json={'question':'total sales by region'})
    assert chat.status_code == 200 and 'Region' in chat.get_json()['sql']
    insights=client.post('/api/insights',headers=headers,json={})
    assert insights.status_code == 200 and len(insights.get_json()['insights']) >= 2
    good=client.post('/api/sql', headers=headers, json={'sql':'SELECT COUNT(*) AS n FROM dataset'})
    assert good.status_code == 200 and good.get_json()['rows'][0][0] == 60
    bad=client.post('/api/sql', headers=headers, json={'sql':'DELETE FROM dataset'})
    assert bad.status_code == 400

def test_cleaning_keeps_original_and_exports_csv(client):
    sid=make_session(client); rows=sample_rows(); rows.append(dict(rows[0]))
    assert upload_sample(client,sid,rows).status_code == 200
    headers={'X-Session-Id':sid}
    clean=client.post('/api/clean',headers=headers,json={'drop_duplicates':True})
    assert clean.status_code == 200 and clean.get_json()['rows_after'] == 60
    profile=client.get('/api/profile',headers=headers).get_json()
    assert profile['profile']['rows'] == 60 and profile['using_cleaned'] is True
    original_sql=client.post('/api/sql',headers=headers,json={'sql':'SELECT COUNT(*) FROM dataset'})
    assert original_sql.get_json()['rows'][0][0] == 61
    reset=client.post('/api/clean/reset',headers=headers)
    assert reset.status_code == 200
    download=client.get('/api/download/cleaned.csv?session_id='+sid)
    assert download.status_code == 200 and b'OrderID' in download.data

def test_file_upload_and_reports(client):
    sid=make_session(client); headers={'X-Session-Id':sid}
    csv=b'Name,Sales\nA,10\nB,20\n'
    uploaded=client.post('/api/upload',headers=headers,data={'file':(BytesIO(csv),'sales.csv')},content_type='multipart/form-data')
    assert uploaded.status_code == 200
    pdf=client.post('/api/report/pdf',headers=headers,json={})
    xlsx=client.post('/api/report/xlsx',headers=headers,json={})
    assert pdf.status_code == 200 and pdf.mimetype == 'application/pdf'
    assert xlsx.status_code == 200 and 'spreadsheetml' in xlsx.mimetype

def test_ml_endpoints(client):
    sid=make_session(client); assert upload_sample(client,sid).status_code == 200
    headers={'X-Session-Id':sid}
    pred=client.post('/api/ml/predict',headers=headers,json={'target':'Sales','features':['Units','UnitPrice','Region']})
    assert pred.status_code == 200 and pred.get_json()['model'] == 'Random Forest Regressor'
    assert {'mae','rmse','r2'} <= set(pred.get_json()['metrics'])
    anomaly=client.post('/api/ml/anomalies',headers=headers,json={'features':['Sales','Profit']})
    assert anomaly.status_code == 200 and 'anomaly_count' in anomaly.get_json()
    segments=client.post('/api/ml/segments',headers=headers,json={'features':['Units','Sales','Profit'],'clusters':3})
    assert segments.status_code == 200 and segments.get_json()['clusters'] == 3
