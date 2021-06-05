const fs = require('fs');
const os = require('os');

const RGRAPH = /^Using graph .*\/(.*?) \.\.\./m;
const RTEMPE = /^Temporal edges: (\d+)/m;
const RBATCH = /^# Batch size ([\d\.e+-]+)/;
const RRESLT = /^order: (\d+) size: (\d+) \{\} \[(.*?) ms; (\d+) iters\.\] \[(.*?) err\.\] (.*)/m;




// *-ARRAY
// -------

function sumArray(x) {
  var a = 0;
  for (var i=0, I=x.length; i<I; i++)
    a += x[i];
  return a;
}

function avgArray(x) {
  return x.length? sumArray(x)/x.length : 0;
}




// *-FILE
// ------

function readFile(pth) {
  var d = fs.readFileSync(pth, 'utf8');
  return d.replace(/\r?\n/g, '\n');
}

function writeFile(pth, d) {
  d = d.replace(/\r?\n/g, os.EOL);
  fs.writeFileSync(pth, d);
}




// *-CSV
// -----

function writeCsv(pth, rows) {
  var cols = Object.keys(rows[0]);
  var a = cols.join()+'\n';
  for (var r of rows)
    a += [...Object.values(r)].map(v => `"${v}"`).join()+'\n';
  writeFile(pth, a);
}




// *-LOG
// -----

function readLogLine(ln, data, state) {
  if (RGRAPH.test(ln)) {
    var [, graph] = RGRAPH.exec(ln);
    if (!data.has(graph)) data.set(graph, []);
    state = {graph};
  }
  else if (RTEMPE.test(ln)) {
    var [, temporal_edges] = RTEMPE.exec(ln);
    state.temporal_edges = parseFloat(temporal_edges);
  }
  else if (RBATCH.test(ln)) {
    var [, batch_size] = RBATCH.exec(ln);
    state.batch_size = parseFloat(batch_size);
  }
  else if (RRESLT.test(ln)) {
    var [, order, size, time, iters, err, technique] = RRESLT.exec(ln);
    data.get(state.graph).push({
      graph: state.graph,
      temporal_edges: state.temporal_edges,
      batch_size: state.batch_size,
      order: parseFloat(order),
      size: parseFloat(size),
      time: parseFloat(time),
      iterations: parseFloat(iters),
      error: parseFloat(err),
      technique
    });
  }
  return state;
}

function readLog(pth) {
  var text = readFile(pth);
  var lines = text.split('\n');
  var data = new Map();
  var state = null;
  for (var ln of lines)
    state = readLogLine(ln, data, state);
  return data;
}




// PROCESS-*
// ---------

function processBatchAverage(rows) {
  var a = [];
  var graph = rows[0].graph;
  var order = Math.max(...rows.map(r => r.order));
  var size = Math.max(...rows.map(r => r.size));
  var batch_size = rows[0].batch_size;
  var techniques = new Set(rows.map(r => r.technique));
  for (var technique of techniques) {
    var rows_filt = rows.filter(r => r.technique===technique);
    var time = avgArray(rows_filt.map(r => r.time));
    var iterations = avgArray(rows_filt.map(r => r.iterations));
    var error = avgArray(rows_filt.map(r => r.error));
    a.push({graph, order, size, batch_size, technique, time, iterations, error});
  }
  return a;
}


function processCsv(data) {
  var a = [];
  for (var rows of data.values())
    a.push(...rows);
  return a;
}


function processShortCsv(data) {
  var a = [];
  for (var rows of data.values()) {
    var batch_sizes = new Set(rows.map(r => r.batch_size));
    for (var batch_size of batch_sizes) {
      var rows_filt = rows.filter(r => r.batch_size===batch_size);
      a.push(...processBatchAverage(rows_filt));
    }
  }
  return a;
}


function processShortLog(data) {
  var a = '';
  for (var rows of data.values()) {
    var order = Math.max(...rows.map(r => r.order));
    var size = Math.max(...rows.map(r => r.size));
    a += `Using graph ${rows[0].graph} ...\n`;
    a += `Temporal edges: ${rows[0].temporal_edges}\n`;
    a += `order: ${order} size: ${size} {}\n\n`;
    var batch_sizes = new Set(rows.map(r => r.batch_size));
    for (var batch_size of batch_sizes) {
      var rows_filt = rows.filter(r => r.batch_size===batch_size);
      a += `# Batch size ${batch_size.toExponential(0)}\n`;
      for (var r of processBatchAverage(rows_filt)) {
        var time = r.time.toFixed(3).padStart(9, '0');
        var iterations = r.iterations.toFixed(0).padStart(3, '0');
        var error = r.error.toExponential(4);
        a += `[${time} ms; ${iterations} iters.] [${error} err.] ${r.technique}\n`;
      }
      a += `\n`;
    }
    a += '\n';
  }
  return a;
}




// MAIN
// ----

function main(cmd, log, out) {
  var data = readLog(log);
  switch (cmd) {
    case 'csv':
      var rows = processCsv(data);
      writeCsv(out, rows);
      break;
    case 'short-csv':
      var rows = processShortCsv(data);
      writeCsv(out, rows);
      break;
    case 'short-log':
      var text = processShortLog(data);
      writeFile(out, text);
      break;
    default:
      console.error(`error: "${cmd}"?`);
      break;
  }
}
main(...process.argv.slice(2));
