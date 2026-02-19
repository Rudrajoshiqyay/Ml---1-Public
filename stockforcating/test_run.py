from prophet_v2 import analyze_stock

if __name__ == '__main__':
    res = analyze_stock('PGEL.NS')
    print('RESULT:', res.get('success'))
    if not res.get('success'):
        print('ERROR:', res.get('error'))
    else:
        print('SUMMARY FILE:', res.get('summary')[:200])
