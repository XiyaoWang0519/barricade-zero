const PLAYER_TYPES = [
  ["human", "Human"],
  ["random", "Random"],
  ["shortest_path", "Shortest path"],
  ["uniform_mcts", "Uniform MCTS"],
  ["checkpoint", "Checkpoint"],
];

const state = {
  game: null,
  checkpoints: [],
  orientation: "H",
  auto: false,
  busy: false,
};

const el = (id) => document.getElementById(id);

function showStatus(message, isError = false) {
  const node = el("status");
  node.textContent = message || "";
  node.classList.toggle("error", Boolean(isError));
}

async function api(path, payload, method) {
  const options = { method: method || (payload === undefined ? "GET" : "POST") };
  if (payload !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(payload);
  }
  const response = await fetch(path, options);
  if (response.status === 204) return null;
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) throw new Error((data && data.error) || response.statusText);
  return data;
}

function fillSelect(select, entries, current) {
  select.innerHTML = "";
  for (const [value, label] of entries) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.append(option);
  }
  if (current) select.value = current;
}

function checkpointOptions() {
  return state.checkpoints
    .filter((item) => item.ok)
    .map((item) => [item.path, `${item.name} · gen ${item.generation ?? "—"} · ${item.board_size}×${item.board_size}`]);
}

function syncCheckpointVisibility() {
  for (const side of [0, 1]) {
    const hide = el(`p${side}-type`).value !== "checkpoint";
    el(`p${side}-checkpoint`).closest("label").hidden = hide;
  }
}

function renderCheckpointList() {
  const list = el("checkpoint-list");
  list.innerHTML = "";
  if (!state.checkpoints.length) {
    list.innerHTML = `<li class="empty">No .pt files in checkpoints/. Train a generation or copy a checkpoint here.</li>`;
    return;
  }
  for (const item of state.checkpoints) {
    const li = document.createElement("li");
    if (item.ok) {
      const button = document.createElement("button");
      button.type = "button";
      button.innerHTML = `<strong>${item.name}</strong><span class="meta">gen ${item.generation ?? "—"} · ${item.channels}ch × ${item.residual_blocks} blocks · ${item.board_size}×${item.board_size}</span>`;
      button.addEventListener("click", () => selectCheckpoint(item));
      li.append(button);
    } else {
      li.innerHTML = `<span class="bad">${item.name}</span><span class="meta">${item.error}</span>`;
    }
    list.append(li);
  }
}

function selectCheckpoint(item) {
  el("board-size").value = item.board_size;
  if (item.walls_per_player != null) el("walls").value = item.walls_per_player;
  el("inspect-checkpoint").value = item.path;
  const side = el("p0-type").value === "checkpoint" ? 0 : 1;
  el(`p${side}-type`).value = "checkpoint";
  el(`p${side}-checkpoint`).value = item.path;
  syncCheckpointVisibility();
}

function playerPayload(side) {
  const type = el(`p${side}-type`).value;
  const payload = { type };
  if (type === "checkpoint") payload.path = el(`p${side}-checkpoint`).value;
  return payload;
}

function currentPlayer() {
  if (!state.game) return null;
  return state.game.players[state.game.state.turn];
}

function isHumanTurn() {
  const game = state.game;
  return Boolean(game && game.live && !game.state.terminal && currentPlayer().type === "human");
}

function lastPly() {
  const game = state.game;
  if (!game || game.viewing_ply <= 0) return null;
  return game.plies[game.viewing_ply - 1];
}

function heatValue(move) {
  if (move.visits) return move.visits;
  if (move.policy) return move.policy;
  if (move.prior) return move.prior;
  return 0;
}

function renderBoard() {
  const root = el("board");
  root.innerHTML = "";
  const game = state.game;
  if (!game) {
    root.innerHTML = `<p class="empty" style="padding:2rem">Start a game to see the board.</p>`;
    return;
  }
  const board = game.state;
  const size = board.size;
  const cell = 72;
  const gap = 14;
  const pad = 22;
  const extent = pad * 2 + size * cell + (size - 1) * gap;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${extent} ${extent}`);
  svg.setAttribute("role", "img");

  const cellOrigin = (index) => pad + index * (cell + gap);
  const last = lastPly();
  const legalPawns = new Map(board.legal.pawns.map((move) => [`${move.row},${move.col}`, move]));
  const maxHeat = Math.max(0, ...board.legal.pawns.map(heatValue), ...board.legal.walls.map(heatValue));

  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      const x = cellOrigin(col);
      const y = cellOrigin(row);
      const key = `${row},${col}`;
      const legal = legalPawns.get(key);
      rect.setAttribute("x", x);
      rect.setAttribute("y", y);
      rect.setAttribute("width", cell);
      rect.setAttribute("height", cell);
      rect.setAttribute("rx", 8);
      rect.classList.add("cell");
      if ((row + col) % 2) rect.classList.add("alt");
      if (legal && isHumanTurn()) {
        rect.addEventListener("click", () => sendMove(legal.action));
      }
      svg.append(rect);
      if (legal && maxHeat && heatValue(legal)) {
        const heat = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        heat.setAttribute("x", x);
        heat.setAttribute("y", y);
        heat.setAttribute("width", cell);
        heat.setAttribute("height", cell);
        heat.setAttribute("rx", 8);
        heat.setAttribute("class", "heat");
        heat.setAttribute("fill", `rgba(212, 119, 78, ${0.12 + 0.45 * (heatValue(legal) / maxHeat)})`);
        svg.append(heat);
      }
    }
  }

  const drawWall = (orientation, row, col, className, parent) => {
    const wall = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    wall.setAttribute("class", `wall ${className || ""}`);
    wall.setAttribute("rx", 4);
    if (orientation === "H") {
      wall.setAttribute("x", cellOrigin(col));
      wall.setAttribute("y", cellOrigin(row) + cell);
      wall.setAttribute("width", cell * 2 + gap);
      wall.setAttribute("height", gap);
    } else {
      wall.setAttribute("x", cellOrigin(col) + cell);
      wall.setAttribute("y", cellOrigin(row));
      wall.setAttribute("width", gap);
      wall.setAttribute("height", cell * 2 + gap);
    }
    (parent || svg).append(wall);
    return wall;
  };

  for (const [row, col] of board.horizontal_walls) drawWall("H", row, col, last?.wall?.orientation === "H" && last.wall.row === row && last.wall.col === col ? "last" : "");
  for (const [row, col] of board.vertical_walls) drawWall("V", row, col, last?.wall?.orientation === "V" && last.wall.row === row && last.wall.col === col ? "last" : "");

  let ghost = null;
  const showGhost = (wall) => {
    if (ghost) ghost.remove();
    ghost = wall ? drawWall(wall.orientation, wall.row, wall.col, "ghost") : null;
  };

  const legalWalls = new Map(
    board.legal.walls.map((wall) => [`${wall.orientation}:${wall.row}:${wall.col}`, wall])
  );
  if (isHumanTurn() && board.walls_remaining[board.turn] > 0) {
    for (let row = 0; row < size - 1; row += 1) {
      for (let col = 0; col < size - 1; col += 1) {
        const hit = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        const x = cellOrigin(col) + cell;
        const y = cellOrigin(row) + cell;
        hit.setAttribute("x", x - 10);
        hit.setAttribute("y", y - 10);
        hit.setAttribute("width", gap + 20);
        hit.setAttribute("height", gap + 20);
        hit.setAttribute("rx", 8);
        hit.setAttribute("class", "anchor");
        const legal = legalWalls.get(`${state.orientation}:${row}:${col}`);
        hit.addEventListener("pointerenter", () => showGhost(legal || null));
        hit.addEventListener("pointerleave", () => showGhost(null));
        if (legal) hit.addEventListener("click", () => sendMove(legal.action));
        svg.append(hit);
      }
    }
  }

  board.pawns.forEach((pawn, index) => {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", cellOrigin(pawn[1]) + cell / 2);
    circle.setAttribute("cy", cellOrigin(pawn[0]) + cell / 2);
    circle.setAttribute("r", cell * 0.28);
    circle.classList.add("pawn", index === 0 ? "p0" : "p1");
    if (!board.terminal && board.turn === index) circle.classList.add("active");
    svg.append(circle);
  });

  root.append(svg);
}

function agentLabel(spec) {
  if (spec.type === "checkpoint") return spec.path;
  return PLAYER_TYPES.find((item) => item[0] === spec.type)?.[1] || spec.type;
}

function renderMatchup() {
  const game = state.game;
  const root = el("matchup");
  if (!game) {
    root.innerHTML = "";
    return;
  }
  root.innerHTML = game.players
    .map((player, index) => {
      const active = !game.state.terminal && game.state.turn === index && game.live;
      return `<article class="match-card ${active ? "active" : ""}">
        <div class="who"><span class="swatch p${index}"></span> Player ${index}</div>
        <span class="meta">${agentLabel(player)}</span>
        <span class="meta">${game.state.walls_remaining[index]} walls · distance ${game.state.distances[index] ?? "—"}</span>
      </article>`;
    })
    .join("");
}

function renderFacts() {
  const game = state.game;
  const root = el("facts");
  if (!game) {
    root.innerHTML = "";
    return;
  }
  const winner = game.state.terminal
    ? game.state.winner == null
      ? "draw"
      : `player ${game.state.winner}`
    : "in play";
  const rows = [
    ["Turn", game.state.terminal ? "—" : `player ${game.state.turn}`],
    ["Result", winner],
    ["Ply", `${game.viewing_ply} / ${game.ply_count}`],
    ["Live", game.live ? "yes" : "viewing history"],
    ["Simulations", game.simulations],
  ];
  root.innerHTML = rows.map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join("");
}

function renderAnalysis() {
  const root = el("analysis");
  const analysis = state.game && state.game.analysis;
  if (!analysis) {
    root.innerHTML = `<p class="hint">Load a checkpoint and inspect the current position, or play an MCTS agent to see visits.</p>`;
    return;
  }
  const value = analysis.value == null ? "—" : analysis.value.toFixed(3);
  const top = (analysis.top || [])
    .map(
      (item) => `<div class="top-move"><span>${item.description}</span><span>${item.visits || 0} visits · ${(item.policy * 100).toFixed(1)}%</span></div>`
    )
    .join("");
  root.innerHTML = `<p class="analysis-value">${value}</p>
    <p class="hint">${analysis.source}${analysis.checkpoint ? ` · ${analysis.checkpoint}` : ""}</p>
    ${top}`;
}

function renderHistory() {
  const root = el("history");
  const game = state.game;
  root.innerHTML = "";
  if (!game) return;
  const initial = document.createElement("li");
  initial.className = game.viewing_ply === 0 ? "current" : "";
  initial.innerHTML = `<button type="button">Start</button>`;
  initial.querySelector("button").addEventListener("click", () => viewPly(0));
  root.append(initial);
  game.plies.forEach((ply, index) => {
    const item = document.createElement("li");
    const plyIndex = index + 1;
    item.className = game.viewing_ply === plyIndex ? "current" : "";
    item.innerHTML = `<button type="button">P${ply.player} ${ply.description}${ply.value == null ? "" : ` <span class="meta">${ply.value.toFixed(2)}</span>`}</button>`;
    item.querySelector("button").addEventListener("click", () => viewPly(plyIndex));
    root.append(item);
  });
}

function renderControls() {
  const game = state.game;
  el("undo").disabled = !game || !game.live || game.ply_count === 0 || state.busy;
  el("reset").disabled = !game || state.busy;
  el("step").disabled = !game || !game.live || game.state.terminal || isHumanTurn() || state.busy;
  el("auto").setAttribute("aria-pressed", String(state.auto));
  el("orient-h").setAttribute("aria-pressed", String(state.orientation === "H"));
  el("orient-v").setAttribute("aria-pressed", String(state.orientation === "V"));
  const max = game ? game.ply_count : 0;
  el("ply").max = String(max);
  el("ply").value = String(game ? game.viewing_ply : 0);
  el("ply-label").textContent = `${game ? game.viewing_ply : 0} / ${max}`;
  const inspectDisabled = !game || !el("inspect-checkpoint").value || state.busy;
  el("inspect-net").disabled = inspectDisabled;
  el("inspect-mcts").disabled = inspectDisabled;
}

function render() {
  renderBoard();
  renderMatchup();
  renderFacts();
  renderAnalysis();
  renderHistory();
  renderControls();
}

function setGame(game) {
  state.game = game;
  render();
  if (game?.state.terminal) {
    const winner = game.state.winner == null ? "Draw" : `Player ${game.state.winner} wins`;
    showStatus(winner);
    state.auto = false;
  } else if (isHumanTurn()) {
    showStatus("Your move — click a highlighted square, or a groove intersection to place a wall.");
  } else if (game) {
    showStatus(`${agentLabel(currentPlayer())} to move.`);
  }
}

async function withBusy(work) {
  state.busy = true;
  renderControls();
  try {
    return await work();
  } finally {
    state.busy = false;
    renderControls();
  }
}

async function newGame() {
  const player0 = playerPayload(0);
  const player1 = playerPayload(1);
  if (player0.type === "checkpoint" && !player0.path) throw new Error("Choose a checkpoint for player 0");
  if (player1.type === "checkpoint" && !player1.path) throw new Error("Choose a checkpoint for player 1");
  const game = await withBusy(() =>
    api("/api/games", {
      board_size: Number(el("board-size").value),
      walls_per_player: Number(el("walls").value),
      simulations: Number(el("simulations").value),
      seed: Number(el("seed").value),
      player0,
      player1,
    })
  );
  state.auto = player0.type !== "human" && player1.type !== "human";
  setGame(game);
  maybeAuto();
}

async function sendMove(action) {
  if (!isHumanTurn() || state.busy) return;
  try {
    const game = await withBusy(() => api(`/api/games/${state.game.id}/move`, { action }));
    setGame(game);
    maybeAuto();
  } catch (error) {
    showStatus(error.message, true);
  }
}

async function step() {
  if (!state.game || state.busy) return;
  const game = await withBusy(() => api(`/api/games/${state.game.id}/step`, {}));
  setGame(game);
  maybeAuto();
}

async function undo() {
  const game = await withBusy(() => api(`/api/games/${state.game.id}/undo`, {}));
  setGame(game);
}

async function resetGame() {
  const game = await withBusy(() => api(`/api/games/${state.game.id}/reset`, {}));
  setGame(game);
  maybeAuto();
}

async function viewPly(ply) {
  const game = await api(`/api/games/${state.game.id}?ply=${ply}`);
  setGame(game);
}

async function inspect(simulations) {
  const checkpoint = el("inspect-checkpoint").value;
  const game = await withBusy(() =>
    api(`/api/games/${state.game.id}/inspect`, {
      checkpoint,
      simulations,
      ply: state.game.viewing_ply,
    })
  );
  setGame(game);
}

function maybeAuto() {
  if (!state.auto || !state.game || !state.game.live || state.game.state.terminal || isHumanTurn() || state.busy) {
    return;
  }
  window.setTimeout(() => {
    step().catch((error) => {
      state.auto = false;
      showStatus(error.message, true);
      renderControls();
    });
  }, Number(el("speed").value));
}

async function loadCheckpoints() {
  const payload = await api("/api/checkpoints");
  state.checkpoints = payload.checkpoints || [];
  const options = checkpointOptions();
  fillSelect(el("p0-checkpoint"), options);
  fillSelect(el("p1-checkpoint"), options);
  fillSelect(el("inspect-checkpoint"), options);
  renderCheckpointList();
  syncCheckpointVisibility();
}

function bind() {
  fillSelect(el("p0-type"), PLAYER_TYPES, "human");
  fillSelect(el("p1-type"), PLAYER_TYPES, "shortest_path");
  el("p0-type").addEventListener("change", syncCheckpointVisibility);
  el("p1-type").addEventListener("change", syncCheckpointVisibility);
  document.querySelectorAll(".presets button").forEach((button) => {
    button.addEventListener("click", () => {
      el("board-size").value = button.dataset.size;
      el("walls").value = button.dataset.walls;
    });
  });
  el("new-game").addEventListener("click", () => newGame().catch((error) => showStatus(error.message, true)));
  el("step").addEventListener("click", () => step().catch((error) => showStatus(error.message, true)));
  el("undo").addEventListener("click", () => undo().catch((error) => showStatus(error.message, true)));
  el("reset").addEventListener("click", () => resetGame().catch((error) => showStatus(error.message, true)));
  el("auto").addEventListener("click", () => {
    state.auto = !state.auto;
    renderControls();
    maybeAuto();
  });
  el("orient-h").addEventListener("click", () => {
    state.orientation = "H";
    renderBoard();
    renderControls();
  });
  el("orient-v").addEventListener("click", () => {
    state.orientation = "V";
    renderBoard();
    renderControls();
  });
  el("ply").addEventListener("input", (event) => viewPly(Number(event.target.value)).catch((error) => showStatus(error.message, true)));
  el("inspect-net").addEventListener("click", () => inspect(0).catch((error) => showStatus(error.message, true)));
  el("inspect-mcts").addEventListener("click", () => inspect(Number(el("simulations").value)).catch((error) => showStatus(error.message, true)));
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea")) return;
    if (event.key === " ") {
      event.preventDefault();
      if (!el("step").disabled) step().catch((error) => showStatus(error.message, true));
    } else if (event.key === "u" || event.key === "U") {
      if (!el("undo").disabled) undo().catch((error) => showStatus(error.message, true));
    } else if (event.key === "h" || event.key === "H") {
      state.orientation = "H";
      renderBoard();
      renderControls();
    } else if (event.key === "v" || event.key === "V") {
      state.orientation = "V";
      renderBoard();
      renderControls();
    }
  });
}

bind();
loadCheckpoints()
  .then(() => showStatus("Choose players and start a game."))
  .catch((error) => showStatus(error.message, true));
render();
