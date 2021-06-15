const fs = require('fs');
const os = require('os');
const path = require('path');

const RGRAPH = /^Loading graph .*\/(.+?)\.mtx \.\.\./m;
const RORDER = /^order: (\d+) size: (\d+) \{\}$/m;
const RMONOL = /^\[(.+?) ms; (\d+) iters\.\] \[(.+?) err\.\] (.+)/m;
const RBATCH = /^# Batch size ([\d\.e+-]+)/m;
const RRESLT = /^order: (\d+) size: (\d+) \{\} \[(.+?) ms; (\d+) iters\.\] \[(.+?) err\.\] (.+)/m;




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
  else if (RORDER.test(ln)) {
    var [, order, size] = RORDER.exec(ln);
    state.original_order = parseFloat(order);
    state.original_size  = parseFloat(size);
  }
  else if (RMONOL.test(ln)) {
    var [, time, iterations, error, technique] = RMONOL.exec(ln);
    state.original_time       = parseFloat(time);
    state.original_iterations = parseFloat(iterations);
    state.original_error      = parseFloat(error);
    state.original_technique  = technique;
  }
  else if (RBATCH.test(ln)) {
    var [, batch_size] = RBATCH.exec(ln);
    state.batch_size = parseFloat(batch_size);
  }
  else if (RRESLT.test(ln)) {
    var [, order, size, time, iterations, error, technique] = RRESLT.exec(ln);
    data.get(state.graph).push(Object.assign({}, state, {
      order:      parseFloat(order),
      size:       parseFloat(size),
      time:       parseFloat(time),
      iterations: parseFloat(iterations),
      error:      parseFloat(error),
      technique
    }));
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
  var techniques = new Set(rows.map(r => r.technique)), a = [];
  for (var technique of techniques) {
    var frows = rows.filter(r => r.technique===technique);
    var order      = avgArray(frows.map(r => r.order));
    var size       = avgArray(frows.map(r => r.size));
    var time       = avgArray(frows.map(r => r.time));
    var iterations = avgArray(frows.map(r => r.iterations));
    var error      = avgArray(frows.map(r => r.error));
    a.push(Object.assign({}, rows[0], {order, size, time, iterations, error, technique}));
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
      var frows = rows.filter(r => r.batch_size===batch_size);
      a.push(...processBatchAverage(frows));
    }
  }
  return a;
}


function processShortLog(data) {
  var a = '';
  for (var rows of data.values()) {
    var r = rows[0];
    var time       = r.time.toFixed(3).padStart(9, '0');
    var iterations = r.iterations.toFixed(0).padStart(3, '0');
    var error      = r.error.toExponential(4);
    a += `Loading graph ${r.graph}.mtx ...\n`;
    a += `order: ${r.order} size: ${r.size} {}\n`;
    a += `[${time} ms; ${iterations} iters.] [${error} err.] ${r.technique}\n\n`;
    var batch_sizes = new Set(rows.map(r => r.batch_size));
    for (var batch_size of batch_sizes) {
      var rows_filt = rows.filter(r => r.batch_size===batch_size);
      a += `# Batch size ${batch_size.toExponential(0)}\n`;
      for (var r of processBatchAverage(rows_filt)) {
        var time       = r.time.toFixed(3).padStart(9, '0');
        var iterations = r.iterations.toFixed(0).padStart(3, '0');
        var error      = r.error.toExponential(4);
        a += `order: ${r.order} size: ${r.size} {} [${time} ms; ${iterations} iters.] [${error} err.] ${r.technique}\n`;
      }
      a += `\n`;
    }
    a += '\n';
  }
  return a.trim()+'\n';
}




// MAIN
// ----

function main(cmd, log, out) {
  var data = readLog(log);
  if (!path.extname(out)) cmd += '-dir';
  switch (cmd) {
    case 'csv':
      var rows = processCsv(data);
      writeCsv(out, rows);
      break;
    case 'csv-dir':
      for (var [graph, rows] of data)
        writeCsv(path.join(out, graph+'.csv'), rows);
      break;
    case 'short-csv':
      var rows = processShortCsv(data);
      writeCsv(out, rows);
      break;
    case 'short-csv-dir':
      for (var [graph, rows] of data) {
        var rows = processShortCsv(new Map([[graph, rows]]));
        writeCsv(path.join(out, graph+'.csv'), rows);
      }
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
