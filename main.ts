// Sokoban solver — original implementation for make.fun "sokoban-solver-01".
//
// Strategy: split the level into independent rooms (cut the player's hallway
// row at its non-live squares, flood-fill the rest), solve each room with a
// push-optimal A* (state = sorted boxes + lowest cell of the player's
// reachable region; min-cost matching heuristic over BFS distances; dead
// squares, statically-stuck cells and frozen 2x2 blocks pruned), then stitch
// the rooms together with walks through the hallway. A whole-level push-A*
// is kept as a fallback if the level does not decompose into rooms.
//
// Deterministic: every budget counts expanded nodes, never wall-clock time;
// no randomness, no network, no file access outside the input/output dirs.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Fixed direction order everywhere (up, down, left, right): all tie-breaks
// in walks, push generation and the heap follow it, so output is stable.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const DL = ['u', 'd', 'l', 'r'];
const INF = 1_000_000_000;
const ROOM_BUDGET_1 = 300_000;
const ROOM_BUDGET_2 = 2_000_000;

type Level = { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number };
type Room = { cells: Uint8Array; boxes: number[]; goals: number[]; door: number; entry: number; minCell: number };
type PushResult = { moves: string; endPlayer: number; finalBoxes: number[] };

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  if (rows.length === 0) return { w: 0, h: 0, wall: new Uint8Array(0), goals: [], boxes: [], player: -1 };
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Live squares: pull-reachability from the goals, ignoring other boxes. A
// cell is live when a box there can be pushed onto a live cell while the
// player stands on the opposite (non-wall) side. Boxes on non-live cells can
// never reach a goal, so pushes landing there are dead by construction.
function liveSquares(lv: Level): Uint8Array {
  const { w, h, wall, goals } = lv;
  const live = new Uint8Array(w * h);
  for (const g of goals) live[g] = 1;
  let changed = true;
  while (changed) {
    changed = false;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const c = y * w + x;
        if (wall[c] || live[c]) continue;
        for (let d = 0; d < 4; d++) {
          const tx = x + DX[d]!, ty = y + DY[d]!;
          const sx = x - DX[d]!, sy = y - DY[d]!;
          if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
          if (!live[ty * w + tx]) continue;
          if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
          if (wall[sy * w + sx]) continue;
          live[c] = 1;
          changed = true;
          break;
        }
      }
    }
  }
  return live;
}

// Split the level into rooms: cut every non-live floor cell of the player's
// row (the hallway), flood-fill what is left. Each component holding boxes
// must hold as many goals as boxes and exactly one door cell (the component
// cell adjacent to a cut hallway cell). Returns null when the level is not a
// clean set of rooms, so the caller can fall back to a whole-level search.
function decompose(lv: Level, live: Uint8Array): Room[] | null {
  const { w, h, wall, goals, boxes, player } = lv;
  const py = (player / w) | 0;
  const cut = new Uint8Array(w * h);
  for (let x = 0; x < w; x++) {
    const c = py * w + x;
    if (!wall[c] && !live[c]) cut[c] = 1;
  }
  for (const b of boxes) if (cut[b]) return null;
  for (const g of goals) if (cut[g]) return null;
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  for (let s = 0; s < w * h; s++) {
    if (wall[s] || cut[s] || seen[s]) continue;
    const cells = new Uint8Array(w * h);
    const stack = [s];
    seen[s] = 1;
    cells[s] = 1;
    let minCell = s;
    while (stack.length > 0) {
      const c = stack.pop()!;
      if (c < minCell) minCell = c;
      const x = c % w, y = (c / w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d]!, ny = y + DY[d]!;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const n = ny * w + nx;
        if (wall[n] || cut[n] || seen[n]) continue;
        seen[n] = 1;
        cells[n] = 1;
        stack.push(n);
      }
    }
    const rb: number[] = [];
    const rg: number[] = [];
    for (const b of boxes) if (cells[b]) rb.push(b);
    for (const g of goals) if (cells[g]) rg.push(g);
    if (rb.length === 0 && rg.length === 0) continue;
    if (rb.length !== rg.length) return null;
    let door = -1, entry = -1, doors = 0;
    for (let c = 0; c < w * h; c++) {
      if (!cells[c]) continue;
      const x = c % w, y = (c / w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d]!, ny = y + DY[d]!;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const n = ny * w + nx;
        if (cut[n]) { door = c; entry = n; doors++; }
      }
    }
    if (doors !== 1) return null;
    rb.sort((a, b) => a - b);
    rg.sort((a, b) => a - b);
    rooms.push({ cells, boxes: rb, goals: rg, door, entry, minCell });
  }
  rooms.sort((a, b) => a.minCell - b.minCell);
  return rooms;
}

// Shortest walk (lowercase moves) from `from` to `to` over `walk` cells,
// treating `blocked` cells as occupied. Directions in fixed [u,d,l,r] order.
function bfsWalk(w: number, h: number, walk: Uint8Array, blocked: Uint8Array, from: number, to: number): string | null {
  if (from === to) return '';
  const prev = new Int32Array(w * h).fill(-2);
  const prevDir = new Int8Array(w * h);
  const q = [from];
  prev[from] = -1;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const x = c % w, y = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = x + DX[d]!, ny = y + DY[d]!;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (!walk[n] || blocked[n] || prev[n] !== -2) continue;
      prev[n] = c;
      prevDir[n] = d;
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) { s = DL[prevDir[cur]!] + s; cur = prev[cur]!; }
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

// Binary heap of [prio1, prio2, seq, stateId]; smallest entry first.
function heapLess(a: number[], b: number[]): boolean {
  return a[0]! < b[0]! || (a[0]! === b[0]! && (a[1]! < b[1]! || (a[1]! === b[1]! && a[2]! < b[2]!)));
}
function heapPush(heap: number[][], e: number[]): void {
  heap.push(e);
  let i = heap.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (!heapLess(heap[i]!, heap[p]!)) break;
    const t = heap[i]!; heap[i] = heap[p]!; heap[p] = t;
    i = p;
  }
}
function heapPop(heap: number[][]): number[] {
  const top = heap[0]!;
  const last = heap.pop()!;
  if (heap.length > 0) {
    heap[0] = last;
    for (let i = 0; ; ) {
      let c = 2 * i + 1;
      if (c >= heap.length) break;
      if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
      if (!heapLess(heap[c]!, heap[i]!)) break;
      const t = heap[i]!; heap[i] = heap[c]!; heap[c] = t;
      i = c;
    }
  }
  return top;
}

// Push-based A* over one closed puzzle (a room, or the whole level as a
// fallback). `walk` = cells the player may stand on; `dest` = cells a box may
// be pushed onto. State = sorted boxes + lowest cell of the player's region.
// With greedy=false this is A* (f = g + h) and the first popped goal state is
// push-optimal; with greedy=true the queue orders by h only, so a solution is
// still found but may use extra pushes. Returns null on budget exhaustion.
function pushSolve(
  w: number, h: number,
  walk: Uint8Array, dest: Uint8Array,
  goals: number[], boxes0: number[], start: number,
  budget: number, greedy: boolean,
): PushResult | null {
  const size = w * h;
  const n = boxes0.length;
  if (n !== goals.length || !walk[start]) return null;
  const goalArr = new Uint8Array(size);
  for (const g of goals) goalArr[g] = 1;
  const startBoxes = boxes0.slice().sort((a, b) => a - b);
  for (const b of startBoxes) if (!dest[b]) return null;

  // BFS distances from each goal over pushable cells.
  const dist: number[][] = [];
  for (const g of goals) {
    const dd = new Array<number>(size).fill(INF);
    dd[g] = 0;
    const q = [g];
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi]!;
      const x = c % w, y = (c / w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d]!, ny = y + DY[d]!;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const m = ny * w + nx;
        if (!dest[m] || dd[m] !== INF) continue;
        dd[m] = dd[c]! + 1;
        q.push(m);
      }
    }
    dist.push(dd);
  }

  // Admissible heuristic: min-cost box->goal matching (exact permutations for
  // n <= 5, sum of per-box minima otherwise). Each push moves one box one
  // step, so the matching cost never overestimates the pushes still needed.
  const hCache = new Map<string, number>();
  let perms: number[][] = [];
  if (n <= 5) {
    const rec = (cur: number[], used: boolean[]): void => {
      if (cur.length === n) { perms.push(cur.slice()); return; }
      for (let i = 0; i < n; i++) {
        if (used[i]) continue;
        used[i] = true;
        cur.push(i);
        rec(cur, used);
        cur.pop();
        used[i] = false;
      }
    };
    rec([], new Array<boolean>(n).fill(false));
  }
  const heuristic = (bs: number[]): number => {
    const key = bs.join(',');
    const hit = hCache.get(key);
    if (hit !== undefined) return hit;
    let best = INF;
    if (n <= 5) {
      for (const p of perms) {
        let s = 0;
        for (let i = 0; i < n; i++) {
          const d = dist[p[i]!]![bs[i]!]!;
          if (d >= INF) { s = INF; break; }
          s += d;
        }
        if (s < best) best = s;
      }
    } else {
      let s = 0;
      for (const b of bs) {
        let m = INF;
        for (let gi = 0; gi < n; gi++) {
          const d = dist[gi]![b]!;
          if (d < m) m = d;
        }
        if (m >= INF) { s = INF; break; }
        s += m;
      }
      best = s;
    }
    hCache.set(key, best);
    return best;
  };

  // Statically stuck cells: a box here can never be pushed again, because in
  // every direction either the landing cell is not pushable or the stance
  // cell is not walkable. (Occupancy by other boxes is transient, so it is
  // not part of this test.) A box on such a cell off-goal is dead.
  const stuck = new Uint8Array(size);
  for (let c = 0; c < size; c++) {
    if (!dest[c] || goalArr[c]) continue;
    const x = c % w, y = (c / w) | 0;
    let can = false;
    for (let d = 0; d < 4; d++) {
      const tx = x + DX[d]!, ty = y + DY[d]!;
      const sx = x - DX[d]!, sy = y - DY[d]!;
      if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
      if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
      if (dest[ty * w + tx] && walk[sy * w + sx]) { can = true; break; }
    }
    if (!can) stuck[c] = 1;
  }

  const maskOf = (bs: number[]): Uint8Array => {
    const m = new Uint8Array(size);
    for (const b of bs) m[b] = 1;
    return m;
  };
  const regionOf = (from: number, bs: number[]): { low: number; mask: Uint8Array } => {
    const occ = maskOf(bs);
    const mask = new Uint8Array(size);
    let low = from;
    const q = [from];
    mask[from] = 1;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi]!;
      if (c < low) low = c;
      const x = c % w, y = (c / w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d]!, ny = y + DY[d]!;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const m = ny * w + nx;
        if (!walk[m] || occ[m] || mask[m]) continue;
        mask[m] = 1;
        q.push(m);
      }
    }
    return { low, mask };
  };
  const cellBlocked = (occ: Uint8Array, cx: number, cy: number): boolean => {
    if (cx < 0 || cx >= w || cy < 0 || cy >= h) return true;
    const cell = cy * w + cx;
    return occ[cell] === 1 || walk[cell] === 0;
  };
  // Frozen 2x2: a block whose four cells are all boxes or unwalkable, holding
  // a box off its goal. No box in it can ever move first (each escape needs a
  // stance or landing cell inside the block, which is occupied), so it is dead.
  const frozenAt = (occ: Uint8Array, tx: number, ty: number): boolean => {
    for (let ox = -1; ox <= 0; ox++) {
      for (let oy = -1; oy <= 0; oy++) {
        const x0 = tx + ox, y0 = ty + oy;
        let all = true;
        let offGoal = false;
        for (let by = 0; by < 2 && all; by++) {
          for (let bx2 = 0; bx2 < 2; bx2++) {
            const cx = x0 + bx2, cy = y0 + by;
            if (!cellBlocked(occ, cx, cy)) { all = false; break; }
            if (cx >= 0 && cx < w && cy >= 0 && cy < h) {
              const cell = cy * w + cx;
              if (occ[cell] === 1 && goalArr[cell] === 0) offGoal = true;
            }
          }
        }
        if (all && offGoal) return true;
      }
    }
    return false;
  };

  for (const b of startBoxes) if (stuck[b] === 1 && goalArr[b] === 0) return null;
  {
    const occ0 = maskOf(startBoxes);
    for (const b of startBoxes) if (frozenAt(occ0, b % w, (b / w) | 0)) return null;
  }
  const h0 = heuristic(startBoxes);
  if (h0 >= INF) return null;

  const bx: number[][] = [startBoxes];
  const pl: number[] = [start];
  const par: number[] = [-1];
  const pf: number[] = [-1];
  const pd: number[] = [-1];
  const gg: number[] = [0];
  const lw: number[] = [regionOf(start, startBoxes).low];
  const best = new Map<string, number>();
  best.set(lw[0] + '|' + startBoxes.join(','), 0);
  const heap: number[][] = [];
  let seq = 0;
  const pushHeap = (p1: number, p2: number, id: number): void => heapPush(heap, [p1, p2, seq++, id]);
  pushHeap(h0, h0, 0);
  let expanded = 0;
  const isGoal = (bs: number[]): boolean => bs.every((b) => goalArr[b] === 1);

  while (heap.length > 0) {
    const s = heapPop(heap)[3]!;
    const skey = lw[s]! + '|' + bx[s]!.join(',');
    if (best.get(skey) !== gg[s]) continue; // stale entry: a better path exists
    const cur = bx[s]!;
    if (isGoal(cur)) {
      const path: number[] = [];
      for (let t = s; t !== -1; t = par[t]!) path.push(t);
      path.reverse();
      let moves = '';
      for (let k = 1; k < path.length; k++) {
        const a = path[k - 1]!, b = path[k]!;
        const from = pf[b]!;
        const dir = pd[b]!;
        const stance = from - (DY[dir]! * w + DX[dir]!);
        const wlk = bfsWalk(w, h, walk, maskOf(bx[a]!), pl[a]!, stance);
        if (wlk === null) return null;
        moves += wlk + DL[dir]!.toUpperCase();
      }
      return { moves, endPlayer: pl[s]!, finalBoxes: cur.slice() };
    }
    if (expanded >= budget) return null;
    expanded++;
    const occ = maskOf(cur);
    const reg = regionOf(pl[s]!, cur);
    for (let i = 0; i < n; i++) {
      const c = cur[i]!;
      const cx = c % w, cy = (c / w) | 0;
      for (let d = 0; d < 4; d++) {
        const tx = cx + DX[d]!, ty = cy + DY[d]!;
        const sx = cx - DX[d]!, sy = cy - DY[d]!;
        if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
        if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
        const t = ty * w + tx, st = sy * w + sx;
        if (!dest[t] || occ[t] === 1) continue; // never push off the room / onto a box
        if (!walk[st] || occ[st] === 1 || reg.mask[st] === 0) continue; // stance unreachable
        const nb = cur.slice();
        nb[i] = t;
        nb.sort((a, b) => a - b);
        if (stuck[t] === 1 && goalArr[t] === 0) continue;
        if (frozenAt(maskOf(nb), tx, ty)) continue;
        const hv = heuristic(nb);
        if (hv >= INF) continue;
        const ng = gg[s]! + 1;
        const reg2 = regionOf(c, nb);
        const key = reg2.low + '|' + nb.join(',');
        const prev = best.get(key);
        if (prev !== undefined && prev <= ng) continue;
        best.set(key, ng);
        const id = bx.length;
        bx.push(nb);
        pl.push(c);
        par.push(s);
        pf.push(c);
        pd.push(d);
        gg.push(ng);
        lw.push(reg2.low);
        pushHeap(greedy ? hv : ng + hv, greedy ? -ng : hv, id);
      }
    }
  }
  return null;
}

// Solve one room with escalating budgets: A* at 300k expansions, A* at 2M,
// then greedy best-first at 2M so a solvable room still finishes (suboptimal)
// instead of scoring nothing.
function solveRoom(
  w: number, h: number, walk: Uint8Array, dest: Uint8Array,
  goals: number[], boxes: number[], entry: number,
): PushResult | null {
  return pushSolve(w, h, walk, dest, goals, boxes, entry, ROOM_BUDGET_1, false)
    ?? pushSolve(w, h, walk, dest, goals, boxes, entry, ROOM_BUDGET_2, false)
    ?? pushSolve(w, h, walk, dest, goals, boxes, entry, ROOM_BUDGET_2, true);
}

function solveByRooms(lv: Level, rooms: Room[], live: Uint8Array): string {
  const { w, h, wall } = lv;
  const floor = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) floor[i] = wall[i] ? 0 : 1;
  const maskOf = (bs: number[]): Uint8Array => {
    const m = new Uint8Array(w * h);
    for (const b of bs) m[b] = 1;
    return m;
  };
  let moves = '';
  let player = lv.player;
  const curBoxes = lv.boxes.slice();
  for (const room of rooms) {
    const toEntry = bfsWalk(w, h, floor, maskOf(curBoxes), player, room.entry);
    if (toEntry === null) throw new Error('cannot walk to room entry');
    moves += toEntry;
    player = room.entry;
    const walk = room.cells.slice();
    walk[room.entry] = 1;
    const dest = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) dest[i] = live[i] && room.cells[i] ? 1 : 0;
    const res = solveRoom(w, h, walk, dest, room.goals, room.boxes, room.entry);
    if (res === null) throw new Error('room has no solution in budget');
    moves += res.moves;
    const exit = bfsWalk(w, h, walk, maskOf(res.finalBoxes), res.endPlayer, room.entry);
    if (exit === null) throw new Error('cannot walk out of room');
    moves += exit;
    player = room.entry;
    const roomSet = new Set(room.boxes);
    const next: number[] = [];
    for (const b of curBoxes) if (!roomSet.has(b)) next.push(b);
    for (const b of res.finalBoxes) next.push(b);
    next.sort((a, b) => a - b);
    curBoxes.splice(0, curBoxes.length, ...next);
  }
  return moves;
}

function solveGlobal(lv: Level, live: Uint8Array): string {
  const { w, h, wall } = lv;
  const floor = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) floor[i] = wall[i] ? 0 : 1;
  const res = pushSolve(w, h, floor, live, lv.goals, lv.boxes, lv.player, ROOM_BUDGET_2, false)
    ?? pushSolve(w, h, floor, live, lv.goals, lv.boxes, lv.player, ROOM_BUDGET_2, true);
  return res === null ? '' : res.moves;
}

function solve(text: string): string {
  const lv = parseLevel(text);
  if (lv.w === 0 || lv.h === 0 || lv.player < 0) return '';
  if (lv.boxes.length === 0) return '';
  const live = liveSquares(lv);
  const rooms = decompose(lv, live);
  if (rooms !== null && rooms.length > 0) {
    try {
      return solveByRooms(lv, rooms, live);
    } catch {
      // A room failed: fall through to the whole-level search.
    }
  }
  return solveGlobal(lv, live);
}

for (const entry of readdirSync(IN_DIR, { withFileTypes: true }).sort((a, b) => (a.name < b.name ? -1 : 1))) {
  if (!entry.isDirectory()) continue;
  const id = entry.name;
  let moves = '';
  try {
    moves = solve(readFileSync(join(IN_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
