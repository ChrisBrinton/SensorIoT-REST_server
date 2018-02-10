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
    return 'hello ' + request.args.get('name', '')


@app.route("/stats", methods=['GET'])
def stats():
    ct = coll.count()
    return 'total rows:' + str(ct)


@app.route('/test', methods=['GET'])
def test():
    return '[49, {"gateway_id": "1A7EFF", "type": "F", "value": 73.24, "time": 1508545497.875178, "node_id": "42", "model": "GDESGW1", "_id": "59ea93d9e13823611d7ecadd"}, {"gateway_id": "1A7EFF", "type": "F", "value": 73.35, "time": 1508547133.091503, "node_id": "42", "model": "GDESGW1", "_id": "59ea9a3de13823611d7ecba0"}, {"gateway_id": "1A7EFF", "type": "F", "value": 73.92, "time": 1508548768.263629, "node_id": "42", "model": "GDESGW1", "_id": "59eaa0a0e13823611d7ecc63"}, {"gateway_id": "1A7EFF", "type": "F", "value": 74.16, "time": 1508550403.443885, "node_id": "42", "model": "GDESGW1", "_id": "59eaa703e13823611d7ecd18"}, {"gateway_id": "1A7EFF", "type": "F", "value": 74.26, "time": 1508552038.684304, "node_id": "42", "model": "GDESGW1", "_id": "59eaad66e13823611d7ecdcc"}, {"gateway_id": "1A7EFF", "type": "F", "value": 73.8, "time": 1508553673.972562, "node_id": "42", "model": "GDESGW1", "_id": "59eab3c9e13823611d7ece99"}, {"gateway_id": "1A7EFF", "type": "F", "value": 72.91, "time": 1508555309.061274, "node_id": "42", "model": "GDESGW1", "_id": "59eaba2de13823611d7ecf5c"}, {"gateway_id": "1A7EFF", "type": "F", "value": 72.43, "time": 1508556943.789801, "node_id": "42", "model": "GDESGW1", "_id": "59eac08fe13823611d7ed015"}, {"gateway_id": "1A7EFF", "type": "F", "value": 72.14, "time": 1508558578.824307, "node_id": "42", "model": "GDESGW1", "_id": "59eac6f2e13823611d7ed0e8"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.92, "time": 1508560213.582492, "node_id": "42", "model": "GDESGW1", "_id": "59eacd55e13823611d7ed1b9"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.76, "time": 1508561848.588553, "node_id": "42", "model": "GDESGW1", "_id": "59ead3b8e13823611d7ed282"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.6, "time": 1508563483.405279, "node_id": "42", "model": "GDESGW1", "_id": "59eada1be13823611d7ed345"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.47, "time": 1508565118.27928, "node_id": "42", "model": "GDESGW1", "_id": "59eae07ee13823611d7ed408"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.35, "time": 1508566752.909039, "node_id": "42", "model": "GDESGW1", "_id": "59eae6e0e13823611d7ed4cb"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.28, "time": 1508568387.441489, "node_id": "42", "model": "GDESGW1", "_id": "59eaed43e13823611d7ed57f"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.17, "time": 1508570021.854898, "node_id": "42", "model": "GDESGW1", "_id": "59eaf3a5e13823611d7ed657"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.08, "time": 1508571656.64864, "node_id": "42", "model": "GDESGW1", "_id": "59eafa08e13823611d7ed715"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.95, "time": 1508573291.072607, "node_id": "42", "model": "GDESGW1", "_id": "59eb006be13823611d7ed7ce"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.84, "time": 1508574925.902762, "node_id": "42", "model": "GDESGW1", "_id": "59eb06cde13823611d7ed891"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.74, "time": 1508576560.54469, "node_id": "42", "model": "GDESGW1", "_id": "59eb0d30e13823611d7ed94b"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.61, "time": 1508578195.26472, "node_id": "42", "model": "GDESGW1", "_id": "59eb1393e13823611d7eda18"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.52, "time": 1508579829.985708, "node_id": "42", "model": "GDESGW1", "_id": "59eb19f5e13823611d7edaef"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.43, "time": 1508581464.664131, "node_id": "42", "model": "GDESGW1", "_id": "59eb2058e13823611d7edbb3"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.36, "time": 1508583098.900616, "node_id": "42", "model": "GDESGW1", "_id": "59eb26bae13823611d7edc7b"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.65, "time": 1508584733.567857, "node_id": "42", "model": "GDESGW1", "_id": "59eb2d1de13823611d7edd44"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.84, "time": 1508586368.452102, "node_id": "42", "model": "GDESGW1", "_id": "59eb3380e13823611d7ede16"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.92, "time": 1508588003.295447, "node_id": "42", "model": "GDESGW1", "_id": "59eb39e3e13823611d7edee3"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.7, "time": 1508589638.011758, "node_id": "42", "model": "GDESGW1", "_id": "59eb4046e13823611d7edfba"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.56, "time": 1508591272.678044, "node_id": "42", "model": "GDESGW1", "_id": "59eb46a8e13823611d7ee088"}, {"gateway_id": "1A7EFF", "type": "F", "value": 70.48, "time": 1508592907.228328, "node_id": "42", "model": "GDESGW1", "_id": "59eb4d0be13823611d7ee148"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.17, "time": 1508594541.812357, "node_id": "42", "model": "GDESGW1", "_id": "59eb536de13823611d7ee206"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.4, "time": 1508596176.739333, "node_id": "42", "model": "GDESGW1", "_id": "59eb59d0e13823611d7ee2bf"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.24, "time": 1508597811.716354, "node_id": "42", "model": "GDESGW1", "_id": "59eb6033e13823611d7ee38c"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.15, "time": 1508599446.728804, "node_id": "42", "model": "GDESGW1", "_id": "59eb6696e13823611d7ee44f"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.1, "time": 1508601081.657891, "node_id": "42", "model": "GDESGW1", "_id": "59eb6cf9e13823611d7ee526"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.11, "time": 1508602716.409735, "node_id": "42", "model": "GDESGW1", "_id": "59eb735ce13823611d7ee5ee"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.15, "time": 1508604351.116862, "node_id": "42", "model": "GDESGW1", "_id": "59eb79bfe13823611d7ee6bb"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.24, "time": 1508605986.027642, "node_id": "42", "model": "GDESGW1", "_id": "59eb8022e13823611d7ee783"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.37, "time": 1508607621.286313, "node_id": "42", "model": "GDESGW1", "_id": "59eb8685e13823611d7ee842"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.44, "time": 1508609256.236783, "node_id": "42", "model": "GDESGW1", "_id": "59eb8ce8e13823611d7ee90f"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.6, "time": 1508610891.314142, "node_id": "42", "model": "GDESGW1", "_id": "59eb934be13823611d7ee9d7"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.64, "time": 1508612526.14946, "node_id": "42", "model": "GDESGW1", "_id": "59eb99aee13823611d7eea9a"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.65, "time": 1508614160.907953, "node_id": "42", "model": "GDESGW1", "_id": "59eba010e13823611d7eeb6c"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.55, "time": 1508615795.801221, "node_id": "42", "model": "GDESGW1", "_id": "59eba673e13823611d7eec39"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.46, "time": 1508617430.949195, "node_id": "42", "model": "GDESGW1", "_id": "59ebacd6e13823611d7eecfc"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.44, "time": 1508619065.603468, "node_id": "42", "model": "GDESGW1", "_id": "59ebb339e13823611d7eedc4"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.4, "time": 1508620700.429771, "node_id": "42", "model": "GDESGW1", "_id": "59ebb99ce13823611d7eee88"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.42, "time": 1508622335.077781, "node_id": "42", "model": "GDESGW1", "_id": "59ebbfffe13823611d7eef46"}, {"gateway_id": "1A7EFF", "type": "F", "value": 71.44, "time": 1508623969.748834, "node_id": "42", "model": "GDESGW1", "_id": "59ebc661e13823611d7ef027"}]'


@app.route('/sensorlist', methods=['GET'])
def sensorlist():
    return json.dumps(coll.distinct('node_id'))


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


def getdata(node, start, skip, type):
    docs = []
    # qry = {'node_id': node, '$and': [{'time': {'$gte': start}}, {'time': {'$lte': end}}]}
    qry = {'node_id': node, 'time': {'$gte': start}}
    sortparam = [('time', 1)]
    if type:
        qry['type'] = type
    print('query is %s and sort is ' % qry, sortparam)
    cursor = coll.find(qry).sort(sortparam)
    ct = 0
    for doc in cursor:
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
    return docs


app.run(host="0.0.0.0", port=5000)

