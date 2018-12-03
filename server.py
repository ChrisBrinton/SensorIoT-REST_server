from flask import Flask
from flask import request
from flask_cors import CORS
import pymongo
from pymongo import MongoClient
import json
import datetime as dt
from dateutil.tz import *

print('connecting to mongo...')
client = MongoClient('localhost', 27017)  # make this explicit
db = client['gdtechdb_prod']
#db = client.test
sensors = db['Sensors']
sensorsLatest = db['SensorsLatest']
app = Flask(__name__)
CORS(app)
timefmt = '%Y-%m-%d %H:%M:%S'

def cleanvalue(value):
    return float(value.replace('b', '').replace('v', '').replace("'", ""))

@app.route("/", methods=['GET'])
def hello():
    r = request
    print(r)
    dt.date.today()
    print('hello returning...')
    return 'hello ' + request.args.get('name', '')


@app.route("/stats", methods=['GET'])
def stats():
    ct = sensors.count()
    return 'total rows:' + str(ct)


@app.route('/sensorlist', methods=['GET'])
def sensorlist():
    print('sensorlist returning...')
    d = sensors.distinct('node_id') 
    print(json.dumps(d))
    return json.dumps(d)


def today():
    now = dt.date.today()
    start = dt.datetime(now.year, now.month, now.day, 0, 0, 0, 0)
    return start.timestamp()


def getstart(p):
    # p should be a number specifying the delta in hours.
    nowdatetime = dt.datetime.now(tzutc())
    if p is None:
        diff = dt.timedelta(hours=24)
    else:
        diff = dt.timedelta(hours=int(p))

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
    try:
        period = int(period)*24
    except ValueError as err:
        period = 24
    start = getstart(period)
    docs = getdata(node, start, skip, type)
    return json.dumps(docs)

@app.route("/nodelist/<gw>", methods=['GET'])
def nodelist(gw):
    period = request.args.get('period')
    try:
        period = int(period)*24
    except TypeError as err:
        period = 24
    start = getstart(period)
    values = getnodelist(gw, start)
    return json.dumps(sorted(values))

def getnodelist(gw, start):
    qry = {'gateway_id': gw, 'time': {'$gte': start}}
    print('query is %s' % qry)
    print('starting query at ', dt.datetime.now())
    values = sensorsLatest.distinct('node_id', qry)
    print('query returned at ', dt.datetime.now()) #this takes 16 millis
    print(json.dumps(sorted(values)))
    return values

@app.route("/latest/<gw>", methods=['GET'])
def latest(gw):
    period = request.args.get('period','')
    try:
        period = int(period)
    except ValueError as err:
        period = 1
    start=getstart(period)
    values = getlatest(gw, start)
    print(json.dumps(values))
    return json.dumps(values)

def getlatest(gw, start):
    docs = []
    qry = {'gateway_id': gw, 'time': {'$gte':int(start)}}
    sortparam = [('node_id', -1)]
    print('query is ', qry, ' sort is ', sortparam)
    cursor = sensorsLatest.find(qry).sort(sortparam)
    print('query run')
    for doc in cursor:
        newdoc = { 'node_id': 0, 'type': '', 'value': 0, 'human_time': '', 'time': 0 }
        newdoc['value'] = float(doc['value'].replace('b', '').replace('v', '').replace("'", ""))
        newdoc['human_time'] = dt.datetime.fromtimestamp(doc['time']).strftime(timefmt)
        newdoc['time'] = doc['time']
        newdoc['type'] = doc['type']
        newdoc['node_id'] = doc['node_id']
        docs.append(newdoc)

    print(json.dumps(docs))
    return docs

@app.route("/gw/<gw>", methods=['GET'])
def gw(gw):
    nodes = request.args.getlist('node')
    type = request.args.get('type', '')
    period = request.args.get('period')
    timezone = request.args.get('timezone','None')
    returndocs = []
    try:
        period = int(period)*24
    except ValueError as err:
        period = 24
    print('calling gwiteratenodes with ', gw, nodes, type, period, timezone, ' at ', dt.datetime.now())
    returndocs = gwiteratenodes(gw, nodes, type, period, timezone)
    return json.dumps(returndocs)

def gwiteratenodes(gw, nodes, type, period, timezone):
    start = getstart(period)
    returndocs = []
    for node in nodes:
        record = {'nodeID': 0, 'sensorData': []}
        record['nodeID'] = node
        print('calling gwdatausinggw with ', gw, node, start, type, timezone, dt.datetime.now())
        record['sensorData'] = getdatausinggw(gw, node, start, type, timezone) 
        print('finished getdatausinggw for', gw, node, start, type, timezone, dt.datetime.now())
        returndocs.append(record)
    return returndocs

def getdatausinggw(gw, node, start, mytype, timezone):
    print('getdatausinggw starting. C extentions in use:', pymongo.has_c())
    docs = []
    resultsarray = []
    empty_results = {'results': '0'}

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
    print('starting query at ', dt.datetime.now())
    cursor = sensors.find(qry).sort(sortparam).batch_size(100000)
    print('query returned at ', dt.datetime.now()) #this takes .2 millis
    for row in cursor:
        resultsarray.append(row)
    print('Arrayified at ', dt.datetime.now())
    count = len(resultsarray)
    print('%i records returned' % count, dt.datetime.now()); #this takes 1.86 sec
    ct = 0
    total = 0
    skip=0
    if count == 0:
        return empty_results
    if count > 300:
        skip = int(count/300 +.49)
        print('Since more than 300 records were returned, skip is set to %i' % skip, dt.datetime.now()) #this take .1 milli

    #insert initial doc as first 'goalpost' with time same as start
    newdoc = {'value': 0, 'human_time': '', 'time': 0}
    # datetimes in DB are utc, convert to local timezone
    newdoc['human_time'] = dt.datetime.fromtimestamp(start).replace(tzinfo=fromZone).astimezone(toZone).strftime(timefmt)
    newdoc['time'] = start
    newvalue = cleanvalue(resultsarray[skip+1]['value'])
    newdoc['value'] = newvalue
    docs.append(newdoc)
    
    for doc in resultsarray:
        total += 1
        ct += 1
        newdoc = {'value': 0, 'human_time': '', 'time': 0}
        # skip if needed
        if ct > skip:
            #newdoc['value'] = float(doc['value'].replace('b', '').replace('v', '').replace("'", ""))
            newvalue = cleanvalue(doc['value'])
            newdoc['value'] = newvalue
            latestvalue = newvalue
            # datetimes in DB are utc, convert to local timezone
            newdoc['human_time'] = dt.datetime.fromtimestamp(doc['time']).replace(tzinfo=fromZone).astimezone(toZone).strftime(timefmt)
            newdoc['time'] = doc['time']
            docs.append(newdoc)
            ct = 0
    if ct !=0:
        newdoc['value'] = float(doc['value'].replace('b', '').replace('v', '').replace("'", ""))
        # datetimes in DB are utc, convert to local timezone
        newdoc['human_time'] = dt.datetime.fromtimestamp(doc['time']).replace(tzinfo=fromZone).astimezone(toZone).strftime(timefmt)
        newdoc['time'] = doc['time']
        docs.append(newdoc)

    #insert last doc as end 'goalpost' with timestamp of now
    newdoc = {'value': 0, 'human_time': '', 'time': 0}
    # datetimes in DB are utc, convert to local timezone
    newdoc['human_time'] = dt.datetime.timestamp(dt.datetime.now())
    newdoc['time'] = dt.datetime.timestamp(dt.datetime.now())
    newdoc['value'] = latestvalue
    docs.append(newdoc)
    
    print('total docs found:', total, ' and returning:', len(docs))
    print('document returned at ', dt.datetime.now()) #this takes 2.28 sec
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
    cursor = sensors.find(qry).sort(sortparam)
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

def testquery():
    docs = []
    qry = {'gateway_id': '16E542'}
    sortparam = [('node_id', -1)]
    cursor = sensorsLatest.find(qry).sort(sortparam)
    for doc in cursor:
        doc['_id'] = str(doc['_id'])  # serialization support
        doc['value'] = float(doc['value'].replace('b', '').replace('v', '').replace("'", ""))
        doc['human_time'] = dt.datetime.fromtimestamp(doc['time']).strftime(timefmt)
        if 'iso_time' in doc:
            doc['iso_time'] = str(doc['iso_time'])
        docs.append(doc)

    print(json.dumps(docs))
    return json.dumps(docs)

def testquery1():
    returndocs = []
    nodes = ['42', '50']
    for node in nodes:
        record = {'nodeID': 0, 'sensorData': []}
        record['nodeID'] = node
        record['sensorData'] = getdatausinggw('16E542', node, 1519344000, 'F', 'EST5EDT') 
        returndocs.append(record)
    print(json.dumps(returndocs))
    return(json.dumps(returndocs))

def testquery2():
    d = getnodelist('16E542',1519344000)

def testquery3():
    period=1
    start=getstart(int(period)*24)
    d = getdatausinggw('16E542','72',start,'F','EST5EDT')
    print(json.dumps(d))

def testquery4():
    start=getstart(1)
    d = getlatest('16E542',start)
    print(json.dumps(d))

def testquery5():
    nodes = ['70', '72']
    returndocs = []
    starttime = dt.datetime.now(tzutc())
    returndocs = gwiteratenodes('16E542', nodes, 'F', 24, 'EST5EDT')
    endtime = dt.datetime.now(tzutc())
    print('executed in ', endtime - starttime)

if __name__ == "__main__":
    values = testquery5() 
    
