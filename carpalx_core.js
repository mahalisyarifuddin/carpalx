const pathCosts = {
  "000": 0.0,
  "001": 0.3,
  "002": 0.6,
  "003": 0.9,
  "006": 1.8,
  "010": 0.3,
  "011": 0.6,
  "012": 0.9,
  "013": 1.2,
  "016": 2.1,
  "020": 0.6,
  "021": 0.9,
  "022": 1.2,
  "023": 1.5,
  "026": 2.4,
  "030": 0.9,
  "032": 1.5,
  "033": 1.8,
  "036": 2.7,
  "040": 1.2,
  "042": 1.8,
  "043": 2.1,
  "046": 3.0,
  "050": 1.5,
  "052": 2.1,
  "053": 2.4,
  "056": 3.3,
  "060": 1.8,
  "062": 2.4,
  "063": 2.7,
  "066": 3.6,
  "070": 2.1,
  "072": 2.7,
  "073": 3.0,
  "076": 3.9,
  "102": 1.6,
  "103": 1.9,
  "104": 2.2,
  "112": 1.9,
  "113": 2.2,
  "114": 2.5,
  "122": 2.2,
  "123": 2.5,
  "124": 2.8,
  "132": 2.5,
  "133": 2.8,
  "134": 3.1,
  "142": 2.8,
  "143": 3.1,
  "144": 3.4,
  "152": 3.1,
  "153": 3.4,
  "154": 3.7,
  "162": 3.4,
  "163": 3.7,
  "164": 4.0,
  "172": 3.7,
  "173": 4.0,
  "174": 4.3,
  "200": 2.0,
  "201": 2.3,
  "202": 2.6,
  "203": 2.9,
  "204": 3.2,
  "205": 3.5,
  "206": 3.8,
  "210": 2.3,
  "211": 2.6,
  "212": 2.9,
  "213": 3.2,
  "214": 3.5,
  "215": 3.8,
  "216": 4.1,
  "217": 4.4,
  "220": 2.6,
  "221": 2.9,
  "222": 3.2,
  "223": 3.5,
  "224": 3.8,
  "225": 4.1,
  "226": 4.4,
  "227": 4.7,
  "230": 2.9,
  "232": 3.5,
  "233": 3.8,
  "234": 4.1,
  "235": 4.4,
  "236": 4.7,
  "237": 5.0,
  "240": 3.2,
  "242": 3.8,
  "243": 4.1,
  "244": 4.4,
  "246": 5.0,
  "247": 5.3,
  "250": 3.5,
  "252": 4.1,
  "253": 4.4,
  "254": 4.7,
  "256": 5.3,
  "257": 5.6,
  "260": 3.8,
  "262": 4.4,
  "263": 4.7,
  "264": 5.0,
  "266": 5.6,
  "267": 5.9,
  "270": 4.1,
  "272": 4.7,
  "273": 5.0,
  "274": 5.3,
  "275": 5.6,
  "276": 5.9,
  "277": 6.2
};

class Carpalx {
    constructor() {
        // Effort component weights
        this.kb = 0.3555;
        this.kp = 0.6423;
        this.ks = 0.4268;

        // Triad interaction parameters
        this.k1 = 1;
        this.k2 = 0.367;
        this.k3 = 0.235;

        // Penalty weights
        this.w0 = 0;
        this.wh = 1;
        this.wr = 1.3088;
        this.wf = 2.5948;

        // Penalties
        this.Ph = { left: 0, right: 0 };
        this.Pr = [1.5, 0.5, 0, 1]; // row 0, 1, 2, 3
        this.Pf = {
            left: [1, 0.5, 0, 0, 0], // pinky to thumb (thumb is index 4)
            right: [0, 0, 0, 0.5, 1] // thumb to pinky (thumb is index 0)
        };

        // Stroke path weights
        this.fh = 1;
        this.fr = 0.3;
        this.ff = 0.3;
        this.path_offset = 0;

        this.baseEfforts = [
            [5, 4, 4, 4, 4, 4, 4.5, 4, 4, 4, 4, 4.5, 5.5],
            [2, 2, 2, 2, 2.5, 3, 2, 2, 2, 2, 2.5, 4, 6],
            [0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 3.5, 2, 2, 2, 2, 2]
        ];

        this.initialLayout = [
            ["`~", "1!", "2@", "3#", "4$", "5%", "6^", "7&", "8*", "9(", "0)", "-_", "=+"],
            ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[{", "]}", "\\|"],
            ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";:", "'\""],
            ["z", "x", "c", "v", "b", "n", "m", ",<", ".>", "/?"]
        ];

        this.fingers = [
            [0, 1, 1, 2, 3, 3, 3, 6, 7, 7, 8, 9, 9],
            [0, 1, 2, 3, 3, 6, 6, 7, 8, 9, 9, 9, 9],
            [0, 1, 2, 3, 3, 6, 6, 7, 8, 9, 9],
            [0, 1, 2, 3, 3, 6, 6, 7, 8, 9]
        ];

        this.mask = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        ];

        this.initKeyboard();
    }

    initKeyboard() {
        this.keys = [];
        this.map = {};
        for (let r = 0; r < this.initialLayout.length; r++) {
            this.keys[r] = [];
            for (let c = 0; c < this.initialLayout[r].length; c++) {
                let keyStr = this.initialLayout[r][c];
                let lc, uc;
                if (keyStr.length === 1 && /[a-zA-Z]/.test(keyStr)) {
                    lc = keyStr.toLowerCase();
                    uc = keyStr.toUpperCase();
                } else if (keyStr.startsWith('\\')) {
                    lc = keyStr[1];
                    uc = keyStr[2] || lc;
                } else {
                    lc = keyStr[0];
                    uc = keyStr[1] || lc;
                }

                let finger = this.fingers[r][c];
                let hand = finger >= 5 ? 1 : 0;

                let key = {
                    row: r,
                    col: c,
                    lc: lc,
                    uc: uc,
                    finger: finger,
                    hand: hand,
                    effort: {}
                };

                this.keys[r][c] = key;
                this.map[lc] = key;
                this.map[uc] = key;

                this.calculateKeyEffort(key);
            }
        }
    }

    calculateKeyEffort(key) {
        let r = key.row;
        let hand = key.hand;
        let finger = key.finger;

        let baseEffort = this.baseEfforts[r][key.col];
        let Ph = hand === 1 ? this.Ph.right : this.Ph.left;
        let Pf = hand === 0 ? this.Pf.left[finger] : this.Pf.right[finger - 5];
        let Pr = this.Pr[r] || 0;

        let penaltyEffort = this.w0 + this.wh * Ph + this.wr * Pr + this.wf * Pf;

        key.effort.base = baseEffort;
        key.effort.penalty = penaltyEffort;
        key.effort.total = this.kb * baseEffort + this.kp * penaltyEffort;
    }

    updateParameters(params) {
        if (params.kb !== undefined) this.kb = params.kb;
        if (params.kp !== undefined) this.kp = params.kp;
        if (params.ks !== undefined) this.ks = params.ks;
        if (params.k1 !== undefined) this.k1 = params.k1;
        if (params.k2 !== undefined) this.k2 = params.k2;
        if (params.k3 !== undefined) this.k3 = params.k3;
        if (params.w0 !== undefined) this.w0 = params.w0;
        if (params.wh !== undefined) this.wh = params.wh;
        if (params.wr !== undefined) this.wr = params.wr;
        if (params.wf !== undefined) this.wf = params.wf;
        if (params.fh !== undefined) this.fh = params.fh;
        if (params.fr !== undefined) this.fr = params.fr;
        if (params.ff !== undefined) this.ff = params.ff;
        if (params.path_offset !== undefined) this.path_offset = params.path_offset;

        if (params.Pr) this.Pr = [...params.Pr];
        if (params.Ph) Object.assign(this.Ph, params.Ph);
        if (params.Pf) {
            if (params.Pf.left) this.Pf.left = [...params.Pf.left];
            if (params.Pf.right) this.Pf.right = [...params.Pf.right];
        }

        for (let r = 0; r < this.keys.length; r++) {
            for (let c = 0; c < this.keys[r].length; c++) {
                this.calculateKeyEffort(this.keys[r][c]);
            }
        }
    }

    calculateEffort(triads) {
        let totalEffort = 0;
        let totalCount = 0;
        for (let triad in triads) {
            let count = triads[triad];
            if (triad.length !== 3) continue;
            let c1 = triad[0], c2 = triad[1], c3 = triad[2];
            if (!this.map[c1] || !this.map[c2] || !this.map[c3]) continue;
            let effort = this.calculateTriadEffort(triad);
            totalEffort += effort * count;
            totalCount += count;
        }
        return totalCount > 0 ? totalEffort / totalCount : 0;
    }

    calculateTriadEffort(triad) {
        let c1 = triad[0], c2 = triad[1], c3 = triad[2];
        let k1 = this.map[c1], k2 = this.map[c2], k3 = this.map[c3];

        let be1 = k1.effort.base, be2 = k2.effort.base, be3 = k3.effort.base;
        let pe1 = k1.effort.penalty, pe2 = k2.effort.penalty, pe3 = k3.effort.penalty;

        let triad_effort = this.kb * this.k1 * be1 * (1 + this.k2 * be2 * (1 + this.k3 * be3)) +
                           this.kp * this.k1 * pe1 * (1 + this.k2 * pe2 * (1 + this.k3 * pe3));

        if (this.ks !== 0) {
            let h1 = k1.hand, h2 = k2.hand, h3 = k3.hand;
            let r1 = k1.row, r2 = k2.row, r3 = k3.row;
            let f1 = k1.finger, f2 = k2.finger, f3 = k3.finger;

            let hand_flag = 0;
            if (h1 === h3) {
                hand_flag = (h2 === h3) ? 2 : 1;
            }

            let finger_flag = 3;
            if (f1 > f2) {
                if (f2 > f3) finger_flag = 0;
                else if (f2 === f3) finger_flag = (c2 === c3) ? 1 : 6;
                else if (f3 === f1) finger_flag = 4;
                else if (f1 > f3 && f3 > f2) finger_flag = 2;
                else finger_flag = 3;
            } else if (f1 < f2) {
                if (f2 < f3) finger_flag = 0;
                else if (f2 === f3) finger_flag = (c2 === c3) ? 1 : 6;
                else if (f3 === f1) finger_flag = 4;
                else if (f1 < f3 && f3 < f2) finger_flag = 2;
                else finger_flag = 3;
            } else if (f1 === f2) {
                if (f2 < f3 || f3 < f1) finger_flag = (c1 === c2) ? 1 : 6;
                else if (f2 === f3) {
                    if (c1 !== c2 && c2 !== c3 && c1 !== c3) finger_flag = 7;
                    else finger_flag = 5;
                }
            }

            let row_flag = 0;
            let r_diff12 = r1 - r2;
            let r_diff13 = r1 - r3;
            let r_diff23 = r2 - r3;
            let diffs = [
                { abs: Math.abs(r_diff12), val: r_diff12 },
                { abs: Math.abs(r_diff13), val: r_diff13 },
                { abs: Math.abs(r_diff23), val: r_diff23 }
            ];
            diffs.sort((a, b) => (b.abs - a.abs) || (a.val - b.val));
            let drmax_abs = diffs[0].abs;
            let drmax = diffs[0].val;

            if (r1 < r2) {
                if (r3 === r2) row_flag = 1;
                else if (r2 < r3) row_flag = 4;
                else if (drmax_abs === 1) row_flag = 3;
                else if (drmax < 0) row_flag = 7;
                else row_flag = 5;
            } else if (r1 > r2) {
                if (r3 === r2) row_flag = 2;
                else if (r2 > r3) row_flag = 6;
                else if (drmax_abs === 1) row_flag = 3;
                else if (drmax < 0) row_flag = 7;
                else row_flag = 5;
            } else {
                if (r2 > r3) row_flag = 2;
                else if (r2 < r3) row_flag = 1;
                else row_flag = 0;
            }

            let path_cost = pathCosts[`${hand_flag}${row_flag}${finger_flag}`];
            if (path_cost === undefined) {
                path_cost = this.fh * hand_flag + this.fr * row_flag + this.ff * finger_flag;
            }
            triad_effort += this.ks * (this.path_offset + path_cost);
        }

        return triad_effort;
    }

    swapKeys(k1_coord, k2_coord) {
        let r1 = k1_coord[0], c1 = k1_coord[1];
        let r2 = k2_coord[0], c2 = k2_coord[1];
        let key1 = this.keys[r1][c1];
        let key2 = this.keys[r2][c2];

        let tmpLc = key1.lc, tmpUc = key1.uc;
        key1.lc = key2.lc; key1.uc = key2.uc;
        key2.lc = tmpLc; key2.uc = tmpUc;

        this.map[key1.lc] = key1; this.map[key1.uc] = key1;
        this.map[key2.lc] = key2; this.map[key2.uc] = key2;

        this.calculateKeyEffort(key1);
        this.calculateKeyEffort(key2);
    }

    getRelocatableKeys() {
        let list = [];
        for (let r = 0; r < this.mask.length; r++) {
            for (let c = 0; c < this.mask[r].length; c++) {
                if (this.mask[r][c]) list.push([r, c]);
            }
        }
        return list;
    }

    findBestSwap(triads) {
        const relocatable = this.getRelocatableKeys();
        let bestSwap = null;
        let bestEffort = this.calculateEffort(triads);

        for (let i = 0; i < relocatable.length; i++) {
            for (let j = i + 1; j < relocatable.length; j++) {
                const k1 = relocatable[i];
                const k2 = relocatable[j];

                this.swapKeys(k1, k2);
                const newEffort = this.calculateEffort(triads);
                if (newEffort < bestEffort) {
                    bestEffort = newEffort;
                    bestSwap = [k1, k2];
                }
                this.swapKeys(k1, k2); // swap back
            }
        }
        return { bestSwap, bestEffort };
    }

    copy() {
        let newCarpalx = new Carpalx();
        newCarpalx.updateParameters({
            kb: this.kb, kp: this.kp, ks: this.ks,
            k1: this.k1, k2: this.k2, k3: this.k3,
            w0: this.w0, wh: this.wh, wr: this.wr, wf: this.wf,
            fh: this.fh, fr: this.fr, ff: this.ff, path_offset: this.path_offset,
            Pr: this.Pr, Ph: this.Ph, Pf: this.Pf
        });
        for (let r = 0; r < this.keys.length; r++) {
            for (let c = 0; c < this.keys[r].length; c++) {
                let key = this.keys[r][c];
                let newKey = newCarpalx.keys[r][c];
                newKey.lc = key.lc;
                newKey.uc = key.uc;
                newCarpalx.map[key.lc] = newKey;
                newCarpalx.map[key.uc] = newKey;
                newCarpalx.calculateKeyEffort(newKey);
            }
        }
        return newCarpalx;
    }
}

if (typeof module !== 'undefined') {
    module.exports = { Carpalx };
}
