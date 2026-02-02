import requests, sys, time
url = 'http://localhost:8000/query'
payload = {'question': 'Hello, can you summarize test.txt?', 'session_id': 'test-session'}
try:
    with requests.post(url, json=payload, stream=True, timeout=15) as r:
        r.raise_for_status()
        start = time.time()
        print('Streaming response:')
        for line in r.iter_lines(decode_unicode=True):
            if line:
                print(line)
            if time.time() - start > 12:
                print('...timed out reading stream')
                break
except Exception as e:
    print('Request failed:', e)
    sys.exit(1)
