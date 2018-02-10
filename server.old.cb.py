from flask import Flask
from flask import request
from flask_cors import CORS
from pymongo import MongoClient
import json
import datetime as dt

print('connecting to mongo...')
client = MongoClient('localhost', 27017)  # make this explicit
db = client['gdtechdb_prod']
#db = client.test
coll = db['Sensors']
app = Flask(__name__)
CORS(app)
timefmt = '%Y-%m-%d %H:%M:%S'


@app.route("/", methods=['GET'])
def hello():
    r = request
    print(r)
    dt.date.today()
    print('hello returning...')
    return 'hello ' + request.args.get('name', '')


@app.route("/stats", methods=['GET'])
def stats():
    ct = coll.count()
    return 'total rows:' + str(ct)


@app.route('/sensorlist', methods=['GET'])
def sensorlist():
    print('sensorlist returning...')
    d = coll.distinct('node_id') 
    print(json.dumps(d))
    return json.dumps(d)


def today():
    now = dt.date.today()
    start = dt.datetime(now.year, now.month, now.day, 0, 0, 0, 0)
    return start.timestamp()


def getstart(p):
    # p should be a number specifying the delta in days.
    now = dt.date.today()
    nowdatetime = dt.datetime(now.year, now.month, now.day, 0, 0, 0, 0)
    if p is None:
        diff = dt.timedelta(days=1)
    else:
        diff = dt.timedelta(days=int(p))

    start = nowdatetime - diff
    return start.timestamp()


@app.route("/sensor/<node>", methods=['GET'])
def people(node):
    skip = request.args.get('skip', '')
    type = request.args.get('type', '')
    period = request.args.get('period')
    try:
        skip = int(skip)
    except ValueError as err:
        print('invalid skip parameter %s. defaulting.' % skip)
        skip = 0
    start = getstart(period)
    docs = getdata(node, start, skip, type)
    return json.dumps(docs)


def getdata(node, start, skip, mytype):
    print('getdata starting...')
    docs = []
    qry = {'node_id': node, 'time': {'$gte': start}}
    #qry = {'node_id': node}
    sortparam = [('time', -1)]
    if mytype:
        qry['type'] = mytype
    print('query is %s and sort is ' % qry, sortparam)
    cursor = coll.find(qry).sort(sortparam)
    print('query run.')  
    ct = 0
    total = 0
    for doc in cursor:
        total += 1
        ct += 1
        # skip if needed
        if ct > skip:
            doc['_id'] = str(doc['_id'])  # serialization support
            doc['value'] = float(doc['value'].replace('b', '').replace('v', '').replace("'", ""))
            doc['human_time'] = dt.datetime.fromtimestamp(doc['time']).strftime(timefmt)
            docs.append(doc)
            ct = 0

    # return number of documents and document list.
    docs.insert(0, docs.__len__())
    print('total docs found:', total, ' and returning:', len(docs))
    return docs

if __name__ == "__main__":
   #app.run(host="0.0.0.0", port=5000)
   d = getdata('54','',0,'F')
   print(json.dumps(d))
    

