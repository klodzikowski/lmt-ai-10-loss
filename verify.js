// verify.js — proves the maths inside index.html is right.
// Extracts the CORE-MATHS block from index.html (so there is no drift between
// what is tested and what runs in the browser) and checks:
//   1. gradients match numerical finite differences for BOTH losses
//   2. both models converge on the embedded 160-passenger subset
//   3. cross-entropy reaches target accuracy in fewer epochs than MSE
//   4. at a confidently-wrong start the MSE gradient has (nearly) vanished
//   5. the subset's survival rates match the published full-dataset rates
// Run: node verify.js
'use strict';
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const START = '// ==== CORE-MATHS-START';
const END = '// ==== CORE-MATHS-END ====';
const iS = html.indexOf(START), iE = html.indexOf(END);
if (iS < 0 || iE < 0) { console.error('FAIL: core-maths markers not found'); process.exit(1); }
const coreSrc = html.slice(html.indexOf('\n', iS) + 1, iE);

const core = new Function(coreSrc +
  '; return { PASSENGERS, LR, MAX_EPOCHS, INITS, buildFeatures, sigmoid, predict, evalLoss, trainEpoch, accuracy };')();
const { PASSENGERS, LR, MAX_EPOCHS, INITS, buildFeatures, predict, evalLoss, trainEpoch, accuracy } = core;

let failures = 0;
function check(name, ok, detail) {
  console.log((ok ? 'PASS' : 'FAIL') + '  ' + name + (detail ? '  —  ' + detail : ''));
  if (!ok) failures++;
}

const { X, y } = buildFeatures(PASSENGERS);
const n = X.length;

// ---------- 0. dataset sanity ----------
check('subset has 160 passengers', n === 160, 'n=' + n);
const rate = idxs => idxs.reduce((s, i) => s + y[i], 0) / idxs.length;
const idxAll = [...Array(n).keys()];
const fem = idxAll.filter(i => PASSENGERS[i][1] === 1);
const male = idxAll.filter(i => PASSENGERS[i][1] === 0);
const rF = rate(fem) * 100, rM = rate(male) * 100, rAll = rate(idxAll) * 100;
// published full-dataset (891) rates: female 74.2%, male 18.9%, overall 38.4%
check('female survival ~74.2%', Math.abs(rF - 74.2) < 4, rF.toFixed(1) + '%');
check('male survival ~18.9%', Math.abs(rM - 18.9) < 4, rM.toFixed(1) + '%');
check('overall survival ~38.4%', Math.abs(rAll - 38.4) < 4, rAll.toFixed(1) + '%');

// ---------- 1. gradient correctness (analytic vs numerical) ----------
// Analytic gradient recovered from trainEpoch's weight update: g = (w_before - w_after) / lr.
// Numerical gradient: central differences on evalLoss.
function analyticGrad(w0, b0, lossType) {
  const m = { w: w0.slice(), b: b0 };
  trainEpoch(m, X, y, 1.0, lossType); // lr = 1 so g = w_before - w_after
  return { gw: w0.map((v, j) => v - m.w[j]), gb: b0 - m.b };
}
function numericalGrad(w0, b0, lossType) {
  const h = 1e-6;
  const gw = w0.map((v, j) => {
    const wp = w0.slice(); wp[j] = v + h;
    const wm = w0.slice(); wm[j] = v - h;
    return (evalLoss({ w: wp, b: b0 }, X, y, lossType) - evalLoss({ w: wm, b: b0 }, X, y, lossType)) / (2 * h);
  });
  const gb = (evalLoss({ w: w0, b: b0 + h }, X, y, lossType) - evalLoss({ w: w0, b: b0 - h }, X, y, lossType)) / (2 * h);
  return { gw, gb };
}
const testPoint = { w: [0.7, -1.3, 0.4, 0.9], b: -0.2 };
for (const lossType of ['ce', 'mse']) {
  const a = analyticGrad(testPoint.w, testPoint.b, lossType);
  const nu = numericalGrad(testPoint.w, testPoint.b, lossType);
  const maxDiff = Math.max(...a.gw.map((g, j) => Math.abs(g - nu.gw[j])), Math.abs(a.gb - nu.gb));
  check('analytic gradient matches numerical (' + lossType.toUpperCase() + ')', maxDiff < 1e-5, 'max diff ' + maxDiff.toExponential(2));
}

// ---------- 2 & 3. convergence + speed comparison ----------
function run(initKey, epochs, targetPct) {
  const out = {};
  for (const lossType of ['ce', 'mse']) {
    const m = { w: INITS[initKey].w.slice(), b: INITS[initKey].b };
    const losses = [evalLoss(m, X, y, lossType)];
    let firstAtTarget = null;
    for (let e = 1; e <= epochs; e++) {
      trainEpoch(m, X, y, LR, lossType);
      losses.push(evalLoss(m, X, y, lossType));
      if (firstAtTarget === null && accuracy(m, X, y) * 100 >= targetPct) firstAtTarget = e;
    }
    let monotone = true;
    for (let e = 1; e < losses.length; e++) if (losses[e] > losses[e - 1] + 1e-12) monotone = false;
    out[lossType] = {
      finalAcc: accuracy(m, X, y) * 100,
      firstLoss: losses[0], lastLoss: losses[losses.length - 1],
      monotone, firstAtTarget,
    };
  }
  return out;
}

// neutral start
const neutral = run('neutral', MAX_EPOCHS, 78);
for (const lt of ['ce', 'mse']) {
  const r = neutral[lt];
  check('converges from neutral start (' + lt.toUpperCase() + '): final accuracy >= 80%', r.finalAcc >= 80, r.finalAcc.toFixed(1) + '%');
  check('loss decreases monotonically (' + lt.toUpperCase() + ')', r.monotone,
    r.firstLoss.toFixed(4) + ' -> ' + r.lastLoss.toFixed(4));
}
check('CE reaches 78% accuracy in <= epochs than MSE (neutral start)',
  neutral.ce.firstAtTarget !== null && neutral.mse.firstAtTarget !== null &&
  neutral.ce.firstAtTarget <= neutral.mse.firstAtTarget,
  'CE: epoch ' + neutral.ce.firstAtTarget + ', MSE: epoch ' + neutral.mse.firstAtTarget);

// confidently-wrong start — the point of the demo
const wrong = run('wrong', 2000, 75);
check('CE recovers from confidently-wrong start', wrong.ce.firstAtTarget !== null && wrong.ce.finalAcc >= 80,
  '75% at epoch ' + wrong.ce.firstAtTarget + ', final ' + wrong.ce.finalAcc.toFixed(1) + '%');
check('MSE eventually recovers too', wrong.mse.firstAtTarget !== null,
  '75% at epoch ' + wrong.mse.firstAtTarget + ', final ' + wrong.mse.finalAcc.toFixed(1) + '%');
check('CE is >= 5x faster than MSE from confidently-wrong start',
  wrong.ce.firstAtTarget !== null && wrong.mse.firstAtTarget !== null &&
  wrong.ce.firstAtTarget * 5 <= wrong.mse.firstAtTarget,
  'CE: ' + wrong.ce.firstAtTarget + ' epochs vs MSE: ' + wrong.mse.firstAtTarget + ' epochs (' +
  (wrong.mse.firstAtTarget / wrong.ce.firstAtTarget).toFixed(1) + 'x slower)');

// ---------- 4. vanishing gradient at confidently-wrong start ----------
function gradNorm(initKey, lossType) {
  const g = analyticGrad(INITS[initKey].w, INITS[initKey].b, lossType);
  return Math.sqrt(g.gw.reduce((s, v) => s + v * v, g.gb * g.gb));
}
const gCe = gradNorm('wrong', 'ce'), gMse = gradNorm('wrong', 'mse');
check('MSE gradient (nearly) vanishes on confident-wrong predictions: CE/MSE gradient ratio > 10',
  gCe / gMse > 10, '|grad| CE=' + gCe.toFixed(4) + ' vs MSE=' + gMse.toFixed(4) + ' (' + (gCe / gMse).toFixed(1) + 'x)');

// ---------- 5. trained predictions are sensible ----------
const trained = { w: INITS.neutral.w.slice(), b: INITS.neutral.b };
for (let e = 0; e < MAX_EPOCHS; e++) trainEpoch(trained, X, y, LR, 'ce');
const byName = name => X[PASSENGERS.findIndex(p => p[0] === name)];
const pChambers = predict(trained.w, trained.b, byName('Mrs. Norman Chambers'));
const pSaunder = predict(trained.w, trained.b, byName('Mr. William Saundercock'));
check('trained CE model: 1st-class woman predicted to survive', pChambers > 0.8, 'p=' + pChambers.toFixed(3));
check('trained CE model: 3rd-class young man predicted not to survive', pSaunder < 0.2, 'p=' + pSaunder.toFixed(3));

console.log('\n' + (failures === 0 ? 'ALL CHECKS PASSED' : failures + ' CHECK(S) FAILED'));
process.exit(failures === 0 ? 0 : 1);
