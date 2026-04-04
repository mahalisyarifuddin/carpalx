const { Carpalx } = require('../carpalx_core.js');
const fs = require('fs');

function getTriads(text) {
    let line = text.replace(/\s/g, '').toLowerCase();
    let triads = {};
    for (let i = 0; i < line.length - 2; i++) {
        let triad = line.substring(i, i + 3);
        triads[triad] = (triads[triad] || 0) + 1;
    }
    return triads;
}

const text = "Thequickbrownfoxjumpsoverthelazydog.Thequickbrownfoxjumpsoverthelazydog.Thequickbrownfoxjumpsoverthelazydog.";
const triads = getTriads(text);

const carpalx = new Carpalx();
const effort = carpalx.calculateEffort(triads);
console.log("Total Effort:", effort);

console.log("Running small optimization...");
const relocatable = carpalx.getRelocatableKeys();
for (let i = 0; i < 100; i++) {
    const k1 = relocatable[Math.floor(Math.random() * relocatable.length)];
    const k2 = relocatable[Math.floor(Math.random() * relocatable.length)];
    carpalx.swapKeys(k1, k2);
    const newEffort = carpalx.calculateEffort(triads);
    if (newEffort < effort) {
        console.log(`Optimization works! Effort reduced from ${effort.toFixed(4)} to ${newEffort.toFixed(4)}`);
        break;
    } else {
        carpalx.swapKeys(k1, k2);
    }
}
