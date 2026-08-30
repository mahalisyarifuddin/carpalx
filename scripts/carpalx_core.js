const fs = require('fs');

class Carpalx {
    constructor() {
        this.kb = 0.3555;
        this.kp = 0.6423;
        this.ks = 0.4268;

        this.k1 = 1;
        this.k2 = 0.367;
        this.k3 = 0.235;

        this.w0 = 0;
        this.wh = 1;
        this.wr = 1.3088;
        this.wf = 2.5948;

        this.Ph = {
            left: 0,
            right: 0
        };
        this.Pr = [1.5, 0.5, 0, 1];
        this.Pf = {
            left: [1, 0.5, 0, 0, 0],
            right: [0, 0, 0, 0.5, 1]
        };

        this.fh = 1;
        this.fr = 0.3;
        this.ff = 0.3;
        this.path_offset = 0;
        this.pathCosts = {};

        this.baseEfforts = [[5, 4, 4, 4, 4, 4, 4.5, 4, 4, 4, 4, 4.5, 5.5], [2, 2, 2, 2, 2.5, 3, 2, 2, 2, 2, 2.5, 4, 6], [0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2], [2, 2, 2, 2, 3.5, 2, 2, 2, 2, 2]];

        this.layouts = {
            qwerty: [["`~", "1!", "2@", "3#", "4$", "5%", "6^", "7&", "8*", "9(", "0)", "-_", "=+"], ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[{", "]}", "\\|"], ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";:", "'\""], ["z", "x", "c", "v", "b", "n", "m", ",<", ".>", "/?"]],
            qwertz: [["`~", "1!", "2@", "3#", "4$", "5%", "6^", "7&", "8*", "9(", "0)", "-_", "=+"], ["q", "w", "e", "r", "t", "z", "u", "i", "o", "p", "[{", "]}", "\\|"], ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";:", "'\""], ["y", "x", "c", "v", "b", "n", "m", ",<", ".>", "/?"]],
            azerty: [["`~", "1!", "2@", "3#", "4$", "5%", "6^", "7&", "8*", "9(", "0)", "-_", "=+"], ["a", "z", "e", "r", "t", "y", "u", "i", "o", "p", "[{", "]}", "\\|"], ["q", "s", "d", "f", "g", "h", "j", "k", "l", "m", "'\""], ["w", "x", "c", "v", "b", "n", ",<", ".>", "/?", ";:"]],
            colemak: [["`~", "1!", "2@", "3#", "4$", "5%", "6^", "7&", "8*", "9(", "0)", "-_", "=+"], ["q", "w", "f", "p", "g", "j", "l", "u", "y", ";:", "[{", "]}", "\\|"], ["a", "r", "s", "t", "d", "h", "n", "e", "i", "o", "'\""], ["z", "x", "c", "v", "b", "k", "m", ",<", ".>", "/?"]],
            dvorak: [["`~", "1!", "2@", "3#", "4$", "5%", "6^", "7&", "8*", "9(", "0)", "[{", "]}"], ["'\"", ",<", ".>", "p", "y", "f", "g", "c", "r", "l", "/?", "=+", "\\|"], ["a", "o", "e", "u", "i", "d", "h", "t", "n", "s", "-_"], [";:", "q", "j", "k", "x", "b", "m", "w", "v", "z"]],
            workman: [["`~", "1!", "2@", "3#", "4$", "5%", "6^", "7&", "8*", "9(", "0)", "-_", "=+"], ["q", "d", "r", "w", "b", "j", "f", "u", "p", ";:", "[{", "]}", "\\|"], ["a", "s", "h", "t", "g", "y", "n", "e", "o", "i", "'\""], ["z", "x", "m", "c", "v", "k", "l", ",<", ".>", "/?"]]
        };
        this.initialLayout = this.layouts.qwerty;

        this.fingers = [[0, 0, 1, 2, 3, 3, 6, 6, 7, 8, 9, 9, 9], [0, 1, 2, 3, 3, 6, 6, 7, 8, 9, 9, 9, 9], [0, 1, 2, 3, 3, 6, 6, 7, 8, 9, 9], [0, 1, 2, 3, 3, 6, 6, 7, 8, 9]];

        this.posTriadEfforts = new Float64Array(0);
        this.initPathCosts();
        this.initKeyboard();
    }

    initPathCosts() {
        this.pathCosts = new Float64Array(192); // 3 * 8 * 8
        for (let h = 0; h <= 2; h++) {
            for (let r = 0; r <= 7; r++) {
                for (let f = 0; f <= 7; f++) {
                    this.pathCosts[h * 64 + r * 8 + f] = this.ks * (this.path_offset + (this.fh * h + this.fr * r + this.ff * f));
                }
            }
        }
    }

    initKeyboard() {
        this.keys = [];
        this.map = {};
        this.positions = [];
        let posId = 0;
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
                    id: posId++,
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
                this.positions.push(key);

                this.calculateKeyEffort(key);
            }
        }
        this.precalculateTriadEfforts();
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

        // Pre-calculate weighted components for triad effort
        key.effort.v1_be = this.kb * this.k1 * baseEffort;
        key.effort.v2_be = this.k2 * baseEffort;
        key.effort.v3_be = this.k3 * baseEffort;
        key.effort.v1_pe = this.kp * this.k1 * penaltyEffort;
        key.effort.v2_pe = this.k2 * penaltyEffort;
        key.effort.v3_pe = this.k3 * penaltyEffort;
    }

    calculateEffort(triads) {
        let totalEffort = 0;
        let totalCount = 0;
        for (let triad in triads) {
            if (triad.length !== 3) continue;
            if (!this.map[triad[0]] || !this.map[triad[1]] || !this.map[triad[2]]) continue;
            let effort = this.calculateTriadEffort(triad);
            totalEffort += effort * triads[triad];
            totalCount += triads[triad];
        }
        return totalCount > 0 ? totalEffort / totalCount : 0;
    }

    precalculateTriadEfforts() {
        const n = this.positions.length;
        if (!this.posTriadEfforts || this.posTriadEfforts.length !== n * n * n) {
            this.posTriadEfforts = new Float64Array(n * n * n);
        }
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                for (let k = 0; k < n; k++) {
                    this.posTriadEfforts[i * n * n + j * n + k] = this._getTriadEffortFromKeys(this.positions[i], this.positions[j], this.positions[k]);
                }
            }
        }
    }

    _getTriadEffortFromKeys(k1, k2, k3) {
        let e1 = k1.effort, e2 = k2.effort, e3 = k3.effort;

        let triad_effort = e1.v1_be * (1 + e2.v2_be * (1 + e3.v3_be)) +
                           e1.v1_pe * (1 + e2.v2_pe * (1 + e3.v3_pe));

        if (this.ks !== 0) {
            let h1 = k1.hand, h2 = k2.hand, h3 = k3.hand;
            let r1 = k1.row, r2 = k2.row, r3 = k3.row;
            let f1 = k1.finger, f2 = k2.finger, f3 = k3.finger;

            let hand_flag = 0;
            if (h1 === h3) {
                hand_flag = (h2 === h3) ? 2 : 1;
            }

            let finger_flag = 3;
            if (f1 === f2) {
                if (f2 === f3) {
                    finger_flag = (k1.id !== k2.id && k2.id !== k3.id && k1.id !== k3.id) ? 7 : 5;
                } else {
                    finger_flag = (k1.id === k2.id) ? 1 : 6;
                }
            } else if (f2 === f3) {
                finger_flag = (k2.id === k3.id) ? 1 : 6;
            } else if (f3 === f1) {
                finger_flag = 4;
            } else if ((f1 < f2 && f2 < f3) || (f1 > f2 && f2 > f3)) {
                finger_flag = 0;
            } else if ((f1 < f3 && f3 < f2) || (f1 > f3 && f3 > f2)) {
                finger_flag = 2;
            }

            let row_flag = 0;
            if (r1 < r2) {
                if (r3 === r2) row_flag = 1;
                else if (r2 < r3) row_flag = 4;
                else {
                    let drmax_abs = Math.abs(r1 - r2), drmax = r1 - r2;
                    let d1 = Math.abs(r1 - r3), v1 = r1 - r3;
                    if (d1 > drmax_abs || (d1 === drmax_abs && v1 < drmax)) { drmax_abs = d1; drmax = v1; }
                    let d2 = Math.abs(r2 - r3), v2 = r2 - r3;
                    if (d2 > drmax_abs || (d2 === drmax_abs && v2 < drmax)) { drmax_abs = d2; drmax = v2; }
                    row_flag = (drmax_abs === 1) ? 3 : ((drmax < 0) ? 7 : 5);
                }
            } else if (r1 > r2) {
                if (r3 === r2) row_flag = 2;
                else if (r2 > r3) row_flag = 6;
                else {
                    let drmax_abs = Math.abs(r1 - r2), drmax = r1 - r2;
                    let d1 = Math.abs(r1 - r3), v1 = r1 - r3;
                    if (d1 > drmax_abs || (d1 === drmax_abs && v1 < drmax)) { drmax_abs = d1; drmax = v1; }
                    let d2 = Math.abs(r2 - r3), v2 = r2 - r3;
                    if (d2 > drmax_abs || (d2 === drmax_abs && v2 < drmax)) { drmax_abs = d2; drmax = v2; }
                    row_flag = (drmax_abs === 1) ? 3 : ((drmax < 0) ? 7 : 5);
                }
            } else {
                if (r2 > r3) row_flag = 2;
                else if (r2 < r3) row_flag = 1;
                else row_flag = 0;
            }

            triad_effort += this.pathCosts[hand_flag * 64 + row_flag * 8 + finger_flag];
        }

        return triad_effort;
    }

    calculateTriadEffort(triad) {
        const k1 = this.map[triad[0]], k2 = this.map[triad[1]], k3 = this.map[triad[2]];
        const n = this.positions.length;
        return this.posTriadEfforts[k1.id * n * n + k2.id * n + k3.id];
    }
}

module.exports = { Carpalx };
