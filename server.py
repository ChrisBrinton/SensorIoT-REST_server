from flask import Flask
from flask import request
from flask_cors import CORS
from pymongo import MongoClient
import json
import datetime as dt
from dateutil.tz import *

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
    nowdatetime = dt.datetime.now(tzutc())
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

@app.route("/gw/<gw>/<int:node>", methods=['GET'])
def gw(gw, node):
    type = request.args.get('type', '')
    period = request.args.get('period')
    timezone = request.args.get('timezone','None')
    skip = 0
    start = getstart(period)
    docs = getdatausinggw(gw, node, start, type, timezone)
    return json.dumps(docs)

def getdatausinggw(gw, node, start, mytype, timezone):
    print('getdatausinggw starting...')
    docs = []

    try:
        toZone = gettz(timezone)
        fromZone = tzutc()
    except ValueError as err:
        print('Invalid timezone parameter %s. Defaulting to 0' % tz)
        tz = 'None'

    qry = {'gateway_id': gw, 'node_id': str(node), 'time': {'$gte': start}}
    sortparam = [('time', 1)]
    if mytype:
        qry['type'] = mytype
    print('query is %s and sort is ' % qry, sortparam)
    count = coll.find(qry).count()
    print('expecting %i records' % count);
    cursor = coll.find(qry).sort(sortparam)
    print('query run.')  
    ct = 0
    total = 0
    skip=0
    if count > 300:
        skip = int(count/300 +.49)
        print('Since more than 300 records were returned, skip is set to %i' % skip)
    for doc in cursor:
        total += 1
        ct += 1
        # skip if needed
        if ct > skip:
            doc['_id'] = str(doc['_id'])  # serialization support
            doc['value'] = float(doc['value'].replace('b', '').replace('v', '').replace("'", ""))
            # datetimes in DB are utc, convert to local timezone
            doc['human_time'] = dt.datetime.fromtimestamp(doc['time']).replace(tzinfo=fromZone).astimezone(toZone).strftime(timefmt)
            if 'iso_time' in doc:
                doc['iso_time'] = str(doc['iso_time'])
            docs.append(doc)
            ct = 0
    if ct !=0:
        doc['_id'] = str(doc['_id'])  # serialization support
        doc['value'] = float(doc['value'].replace('b', '').replace('v', '').replace("'", ""))
        doc['human_time'] = dt.datetime.fromtimestamp(doc['time']).replace(tzinfo=fromZone).astimezone(toZone).strftime(timefmt)
        if 'iso_time' in doc:
            doc['iso_time'] = str(doc['iso_time'])
        docs.append(doc)

    format_hour = "%I:%M%p"
    format_day = "%b %d"
    numofpips = 7
    xvalues = dict()
    for i in range(numofpips-1):
        chunk = int((len(docs)-1)/(numofpips-1))
        datestr = dt.datetime.fromtimestamp(docs[chunk*i]['time']).replace(tzinfo=fromZone).astimezone(toZone).strftime(format_day) 
        if datestr[0] == '0':
            datestr = datestr[1:]
        timestr = dt.datetime.fromtimestamp(docs[chunk*i]['time']).replace(tzinfo=fromZone).astimezone(toZone).strftime(format_hour)
        if timestr[0] == '0':
            timestr = timestr[1:]
        xvalues[i] = {
        'date': datestr,
        'time': timestr
        }
    datestr = dt.datetime.fromtimestamp(docs[len(docs)-1]['time']).replace(tzinfo=fromZone).astimezone(toZone).strftime(format_day) 
    if datestr[0] == '0':
        datestr = datestr[1:]
    timestr = dt.datetime.fromtimestamp(docs[len(docs)-1]['time']).replace(tzinfo=fromZone).astimezone(toZone).strftime(format_hour)
    if timestr[0] == '0':
        timestr = timestr[1:]
    xvalues[numofpips] = {
    'date': datestr,
    'time': timestr
    }
    docs.insert(0, xvalues)
    print('total docs found:', total, ' and returning:', len(docs))
    return docs

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
            if 'iso_time' in doc:
                doc['iso_time'] = str(doc['iso_time'])
            docs.append(doc)
            ct = 0

    # return number of documents and document list.
    docs.insert(0, docs.__len__())
    print('total docs found:', total, ' and returning:', len(docs))
    return docs

if __name__ == "__main__":
   #app.run(host="0.0.0.0", port=5000)
   d = getdatausinggw('16E542',42,1519344000,'F','EST5EDT')
   print(json.dumps(d))
    

