const { Carpalx } = require('./carpalx_core.js');
const fs = require('fs');

const carpalx = new Carpalx();
const text = fs.readFileSync('scripts/test_corpus.txt', 'utf8');

function getTriads(text) {
    let triads = {};
    const lines = text.split(/\r?\n/);
    for (let line of lines) {
        line = line.toLowerCase().replace(/[^a-z]/g, '');
        if (line.length < 3) continue;
        for (let i = 0; i < line.length - 2; i++) {
            let triad = line.substring(i, i + 3);
            if (triad[0] === triad[1] && triad[1] === triad[2]) continue;
            triads[triad] = (triads[triad] || 0) + 1;
        }
    }
    return triads;
}

const triads = getTriads(text);
const effort = carpalx.calculateEffort(triads);
console.log(`Effort: ${effort.toFixed(6)}`);
