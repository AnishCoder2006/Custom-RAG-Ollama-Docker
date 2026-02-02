#!/usr/bin/env python
import sys
import os

# Add data/backend to path so 'app' module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data', 'backend'))

import uvicorn
from app.main import app

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8888)
