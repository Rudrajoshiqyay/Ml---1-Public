import json
import traceback
from prophet_v2 import analyze_stock

try:
    res = analyze_stock('PGEL.NS')
    out = {
        'success': bool(res.get('success')),
        'patterns': res.get('patterns'),
        'pattern_name': res.get('patterns').get('name') if isinstance(res.get('patterns'), dict) else None,
        'pattern_info': res.get('patterns')
    }
    print(json.dumps(out, default=str))
except Exception as e:
    print(json.dumps({'success': False, 'error': str(e), 'trace': traceback.format_exc()}))
